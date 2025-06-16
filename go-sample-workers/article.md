# Go along the Pipeline: Sizer, Hasher, Matcher

## Anatomy of Workers in a File Duplicate Detector.

---

Previous articles:
- [Part #1. Building a Queue for Go Pipelines](https://dev.to/andrey_matveyev/building-a-queue-for-go-pipelines-24b)
- [Part #2. Parallel Traversal of Recursive Structures in Go.](https://dev.to/andrey_matveyev/parallel-traversal-of-recursive-structures-in-go-12kn)

Let's imagine we set ourselves the task of creating a **file duplicate detector**. How would we approach this problem? Most likely, we would need a pipeline of specialized components, each performing its unique part of the work. In our previous articles, we laid the groundwork for building a file processing pipeline capable of handling large volumes of data, and we discussed the advantages of a modular approach and concurrent programming. Today, it's time to dive deeper into the heart of our pipeline - its **workers** - specialized components that perform the core work of detecting duplicates.

These workers are organized into sequential stages, each performing its unique task, passing the results to the next, more detailed stage. This design allows for efficient filtering of unique files at early stages, minimizing the load on subsequent, more resource-intensive steps.

### Unit of Work: The `task` Structure

Before we delve into the workers, it's important to understand the information they operate on. Not just files, but their descriptions, encapsulated in the `task` structure, are passed along the pipeline between workers. In previous versions of our system, the `task` structure was simpler, but as the duplicate detector's functionality evolved, it expanded to carry all the necessary information for decision-making at various stages.

The current form of the `task` structure in our package looks like this:

```Go
// key contains the main file attributes used for identification and grouping.
type key struct {
	size  int64 // File size
	hash  uint32 // File hash (e.g., CRC32)
	equal int // Comparison counter (how many times the file was part of a potential duplicate group)
}

// info contains file metadata not directly used for comparison but important for processing.
type info struct {
	checked bool // Flag indicating whether the file has already been checked
	path    string // Full path to the file
}

// task is the main unit of work passed along the pipeline.
// It combines key and info
type task struct {
	key
	info
}

// newTask creates a new task instance.
func newTask(key key, info info) *task {
	return &task{key: key, info: info}
}
```

### Checker: Guardian of Uniqueness

Before we look at specific workers, it's important to mention the **`checker`** component. It is not a worker itself but plays a critically important role in the operation of `sizer`, `hasher`, and `matcher`. The `checker` is a mechanism that helps avoid reprocessing already checked files or comparing a file to itself. It stores information about files that have already been "registered" at previous stages and helps determine whether the current file is a potential duplicate requiring further verification, or if it has already been processed. This ensures efficient filtering and reduces unnecessary work.

The implementation of `checker` looks like this:

```Go
package fdd

import "sync"

// checker defines the interface for checking and reviewing tasks.
type checker interface {
	verify(task *task) (checkedTask *task, detected bool) // Checks if a task has already been processed or is part of duplicates.
	review(task *task) (checkedTask *task)               // Reviews tasks for comparison with the current one.
}

// checkList - a concrete implementation of checker using a map to store information.
type checkList struct {
	mtx  sync.Mutex      // Mutex for safe access to the map from multiple goroutines.
	list map[key]info // Map where the key is the task key, and the value is info.
}

// newCheckList creates a new checker instance.
func newCheckList() checker {
	return &checkList{list: make(map[key]info)}
}

// verify checks if a task has been previously added to the list.
// Returns (task, true) if a duplicate is found that has not yet been marked as checked,
// (nil, true) if a duplicate is found and already checked, (nil, false) if no duplicate is found.
func (item *checkList) verify(task *task) (checkedTask *task, detected bool) {
	item.mtx.Lock()
	defer item.mtx.Unlock()

	info, detected := item.list[task.key]
	if detected {
		if info.checked { // If already checked, return nil
			return nil, detected
		}
		checkedTask = newTask(task.key, info) // Create a task to return
		info.checked = true                   // Mark as checked
		item.list[task.key] = info
		return checkedTask, detected
	}
	item.list[task.key] = task.info // If not found, add to the list
	return nil, detected
}

// review returns a task from the list for comparison with the current one, if it exists.
// Used to find the next candidate for byte-by-byte comparison.
func (item *checkList) review(task *task) (checkedTask *task) {
	item.mtx.Lock()
	defer item.mtx.Unlock()

	info, detected := item.list[task.key]
	if detected {
		return newTask(task.key, info) // Return the found task
	}
	item.list[task.key] = task.info // If not found, add to the list
	return nil
}
```

### Sizer: The First Sieve

Let's start with the simplest, yet highly effective worker—the **`Sizer`**.

* **What it does:** Its task is to determine the size of each file.

* **Why it's important:** This is the first and fastest filtering stage. Two files cannot be duplicates if they have different sizes. The `Sizer` quickly discards files with unique sizes, sending only those with matching sizes further down the pipeline. Size information is extracted directly from file metadata (e.g., `os.FileInfo`), making this process very fast and not requiring reading file content.

Implementation of the `Sizer` worker:

```Go
// sizer is a worker that processes tasks by determining file size.
type sizer struct{}

// run is the main execution loop for the sizer worker.
// It reads tasks from the inp channel, checks their size (which is already known at this stage),
// and sends them to the out channel if they meet the criteria (not filtered).
func (item *sizer) run(inp, out chan *task, checker checker) {
	for currentTask := range inp {
		// Sizer logic is simple: it already has the size from the previous stage
		// and passes it to the next stage. No additional IO or blocking for size.
		checkedTask, detected := checker.verify(currentTask)
		if detected {
			out <- currentTask
			if checkedTask != nil {
				out <- checkedTask
			}
		}
	}
}
```

### Hasher: The Second Sieve (More Refined)

The next step in our pipeline is the **`Hasher`** worker.

* **What it does:** The `Hasher` is responsible for calculating the checksum (hash) of the file content. In our case, this could be, for example, CRC32.

* **Why it's more complex:** Unlike the `Sizer`, the `Hasher` **must read part or all of the file content** to calculate the hash. This makes it more resource-intensive. However, it only operates on files that have passed the `Sizer` stage (i.e., have the same size).

* **Role in the pipeline:** The `Hasher` performs a more precise filtering. If two files have the same size but different hashes, they are definitely not duplicates. Only files with the same size and the same hash proceed to the next, most resource-intensive stage.

Implementation of the `Hasher` worker:

```Go
// hasher is a worker that calculates the hash of files.
type hasher struct{}

// run is the main execution loop for the hasher worker.
// It reads tasks from the inp channel, opens the file, reads its content
// (or part of it) to calculate the hash, and then sends the task further.
func (item *hasher) run(inp, out chan *task, checker checker) {
	buf := make([]byte, 512) // Buffer for reading part of the file
	
	for inpTask := range inp {
		func(currentTask *task) { // Anonymous function for using defer
			file, err := os.Open(currentTask.path)
			// checkError logs the error if it's not nil and not io.EOF.
			// nil in the last parameter, as it's not relevant here.
			if checkError(err, "File open error.", "os.Open()", item, currentTask, nil) {
				return // Skip the task on open error
			}
			defer func(f *os.File) {
				closeErr := f.Close()
				// checkError logs the file close error.
				checkError(closeErr, "File close error.", "file.Close()", item, currentTask, nil)
			}(file)

			n, err := file.Read(buf) // Read part of the file for hashing
			// file will be ignored, if size=0 or Read returns a non-EOF error
			// checkError logs the file read error.
			if checkError(err, "File read error.", "file.Read()", item, currentTask, nil) {
				return // Skip the task on read error (excluding EOF)
			}

			currentTask.key.hash = crc32.ChecksumIEEE(buf[:n]) // Calculate hash
			checkedTask, detected := checker.verify(currentTask)
			if detected {
				out <- currentTask // Send the current task
				if checkedTask != nil {
					out <- checkedTask // Also send the previously found duplicate
				}
			}
		}(inpTask)
	}
}
```

### Matcher: Final Verification

And finally, the most complex and resource-intensive worker—the **`Matcher`**.

* **What it does:** The `Matcher` performs a byte-by-byte comparison of files. If the `Sizer` and `Hasher` have determined that two files have the same size and the same hash, the `Matcher` conducts a final, unambiguous check by comparing their content byte by byte.

* **Why it's the most complex stage:** Byte-by-byte comparison requires **reading the entire content of both files**, which is the most expensive operation in terms of time and I/O resources.

* **Role in the pipeline:** This stage is the ultimate verification. Only after successfully passing the `Matcher` are files recognized as true duplicates. Thanks to the previous filters (`Sizer` and `Hasher`), the `Matcher` processes significantly fewer files, which is critically important for performance.

Implementation of the `Matcher` worker:

```Go
// matcher is a worker that performs byte-by-byte comparison of files to confirm duplicates.
type matcher struct{}

// run is the main execution loop for the matcher worker.
// It reads tasks from the inp channel and compares them with duplicate candidates using the checker.
func (item *matcher) run(inp, out chan *task, checker checker) {
	buf1 := make([]byte, 2*1024) // Buffer for reading data from the first file
	buf2 := make([]byte, 2*1024) // Buffer for reading data from the second file
	
	// Main loop: reads tasks from the input channel inp.
	for inpTask := range inp {
		// Anonymous function to encapsulate defer and control break/continue in the inner loop.
		func(currentTask *task) {
			for { // Inner loop to compare currentTask with all its potential duplicates.
				// Get the next candidate for byte-by-byte comparison from the checker.
				reviewedTask := checker.review(currentTask) 
				if reviewedTask == nil {
					// If there are no more candidates for the current task, exit the inner loop.
					break 
				}

				// Opening the first file for comparison.
				file1, err := os.Open(currentTask.path)
				// checkError logs the file open error and returns true if the current task needs to be interrupted.
				// nil in the last parameter, as reviewedTask is not used here for logging the path.
				if checkError(err, "File1 open error.", "os.Open()", item, currentTask, nil) {
					// If there's an error opening file1, skip the current task and exit the inner loop.
					break 
				}
				defer func() {
					// Closing file1 if it was successfully opened.
					if file1 != nil {
						closeErr := file1.Close()
						// checkError logs the file1 close error.
						checkError(closeErr, "File1 close error.", "file1.Close()", item, currentTask, nil)
					}
				}()

				// Opening the second file for comparison.
				file2, err := os.Open(reviewedTask.path)
				// checkError logs the file2 open error and returns true if the current task needs to be skipped.
				// reviewedTask is passed as context for the error if it occurred with the second file.
				if checkError(err, "File2 open error.", "os.Open()", item, reviewedTask, nil) {
					// Increment the comparison counter for currentTask, as comparison with this reviewedTask failed.
					currentTask.key.equal++ 
					// If file2 cannot be opened, we still want to continue comparing currentTask
					// with other reviewedTask candidates, if any.
					// Rewind file1 to the beginning to prepare for the next comparison.
					file1.Seek(0, io.SeekStart)
					continue // Proceed to the next reviewedTask candidate.
				}
				defer func() {
					// Closing file2 if it was successfully opened.
					if file2 != nil {
						closeErr := file2.Close()
						// checkError logs the file2 close error.
						checkError(closeErr, "File2 close error.", "file2.Close()", item, reviewedTask, nil)
					}
				}()

				// Perform byte-by-byte comparison of the two files.
				filesEqual, checkErr := checkEqual(file1, file2, buf1, buf2)
				// Special handling for checkEqual error. Logged at Info level, as requested.
				// This may indicate issues reading from files during comparison.
				// currentTask and reviewedTask are passed for maximum information in the log.
				checkError(checkErr, "Check equal error (file1.Read() or file2.Read()).", "checkEqual()", item, currentTask, reviewedTask)

				if filesEqual {
					// If the files are found to be byte-for-byte equal.
					verifiedTask, detected := checker.verify(currentTask)
					if detected {
						out <- currentTask // Send the current file as a confirmed duplicate.
						if verifiedTask != nil {
							out <- verifiedTask // Also send the previously found duplicate, which is now confirmed.
						}
						break // A match is found; no need to compare currentTask with other candidates.
					}
					currentTask.key.equal++ // Increment the comparison counter.
					// Rewind file1 for the next comparison (if the current task is not yet fully verified,
					// i.e., it still needs comparisons for final duplicate group determination).
					file1.Seek(0, io.SeekStart)
					continue // Proceed to the next candidate.
				} else {
					// If the files are not byte-for-byte equal.
					currentTask.key.equal++ // Increment the comparison counter.
					// Rewind file1 for the next comparison (with another candidate).
					file1.Seek(0, io.SeekStart)
					continue // Proceed to the next candidate.
				}
			}
		}(inpTask)
	}
}

// Byte-to-byte compare two files
func checkEqual(file1, file2 io.Reader, buf1, buf2 []byte) (bool, error) {
	for {
		n1, err1 := file1.Read(buf1)
		n2, err2 := file2.Read(buf2)

		if err1 == io.EOF && err2 == io.EOF {
			return true, nil
		}

		if err1 == io.EOF || err2 == io.EOF {
			return false, nil
		}

		if err1 != nil {
			return false, err1
		}
		if err2 != nil {
			return false, err2
		}

		if n1 != n2 {
			return false, nil
		}

		if !bytes.Equal(buf1[:n1], buf2[:n2]) {
			return false, nil
		}
	}
}
```

### Conclusion

Thus, each worker in our pipeline performs a specialized task, contributing to efficient and multi-stage file filtering. From simple size determination to complex byte-by-byte comparison, these components work in unison to quickly and accurately find duplicates. The `checker`, in turn, ensures the correctness and optimization of this process at every step.

In the next article, we will combine all these components, as well as other elements such as queues and the dispatcher, to assemble a complete **File Duplicate Detector (FDD)** and demonstrate its full functionality.