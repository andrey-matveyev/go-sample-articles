# Parallel Traversal of Recursive Structures in Go.

## A Flexible Approach Without Stack Issues

---

### Introduction

In the world of programming, we often encounter the task of processing data represented as recursive or hierarchical structures – be it a file system, XML/JSON trees, graph data structures, or object hierarchies. Traditional recursive traversal, while intuitive, has its limitations, especially in high-load systems or very deep structures, where it can lead to `stack overflow`. Moreover, synchronous traversal often inefficiently utilizes computational resources, particularly with I/O-bound operations.

In Go, with its powerful concurrency model based on goroutines and channels, we can implement an elegant and scalable approach to parallel processing of recursive structures that **completely eliminates stack depth issues** and efficiently parallelizes tasks.

In this article, we will delve into the architecture of such a solution. To manage the flow of tasks between parallel processors, we will utilize a flexible queue, which you can read more about in my previous [publication](https://dev.to/andrey_matveyev/building-a-queue-for-go-pipelines-24b). As a practical example, we will implement a parallel file system traversal to output a list of files and their sizes from a specified directory of arbitrary depth.

### The Challenge of Traditional Recursion and the Go Solution

Classic recursive calls, where a function calls itself to process subtasks, are well-suited for structures with shallow depth. However, if the recursion depth becomes too great (e.g., when traversing a file system with thousands of nested directories), the operating system might allocate insufficient memory for the call stack, leading to a `stack overflow` error and program termination.

Our Go solution offers an alternative: instead of each goroutine recursively calling another function that, in turn, would spawn a new goroutine, we transform "recursive calls" into **messages passed through channels**. These messages represent new tasks (e.g., paths to subdirectories) that are added to a central queue. A pool of "workers" (goroutines) then concurrently extracts tasks from this queue, processes them, and, if necessary, generates new "recursive" tasks for subsequent processing.

This approach provides:
* **Elimination of `stack overflow`**: Since goroutines do not make deep recursive calls into the stack, the stack overflow problem disappears.
* **Arbitrary traversal depth**: The system is capable of processing structures of any nesting level.
* **Scalability**: The number of parallel workers is easily adjustable, allowing for efficient utilization of available CPUs and resources.
* **Efficient I/O management**: I/O-bound tasks (e.g., reading directories) can be performed concurrently without blocking the entire process.

### Solution Architecture

Our solution consists of three main components interacting with each other via Go channels and managed by `context.Context` for proper shutdown.

1.  **Producer/Initiator (`worker.Start`)**: Responsible for launching the initial task (root directory) and managing the overall count of pending "recursive" tasks (`sync.WaitGroup`). It also handles closing the input channel for the queue when all tasks are processed.

2.  **Queue (`queue` package)**: Acts as a buffer between task producers (in this case, the initiator itself and workers finding new directories) and consumers (the worker pool). It decouples the rate of task production from the rate of consumption, preventing `busy-waiting` and overloads. A detailed description of the queue implementation is available at [Building a Queue for Go Pipelines](https://dev.to/andrey_matveyev/building-a-queue-for-go-pipelines-24b).

3.  **Worker Pool (`worker.RunPool` and `worker.Worker`)**: A set of goroutines that concurrently extract tasks from the queue, process them (e.g., read directory contents), and:
    * If a file is found — send its information to the output channel.
    * If a subdirectory is found — add a new "recursive" task to a channel that is fed back into the queue.

The architectural diagram you can see in the main picture of the publication.

### Implementation: File System Traversal

Let's look at the key code parts that implement this architecture using a file system traversal as an example.

#### Task Structure (`queue.Task`)

To pass information about files and directories between components, we use a simple `Task` structure:

```Go
type Task struct {
	Size int64  // File size (0 for directories)
	Path string // Full path to the file or directory
}
```

#### Initiator and Completion Management (`worker.Start`)

The `worker.Start` function kicks off the entire traversal process. It performs two key actions:
1.  Initializes a `sync.WaitGroup` (`recCount`), adding a counter for the initial directory.
2.  Starts a goroutine that waits until `recCount` becomes zero (i.e., all recursive tasks are processed) or wait signal `sync.Cond`(`recClose`) about all workers was stoped (if context `ctx` was canceled)
`worker.Start` closes the channel `rec` that is fed into the queue. This signals the complete end of the traversal.

```Go
var recCount sync.WaitGroup
var recClose = sync.NewCond(&sync.Mutex{})

func Start(path string) chan *queue.Task {
	rec := make(chan *queue.Task)

	// send first task to "rec" Chan
	recCount.Add(1)
	go func() {
		rec <- &queue.Task{Size: 0, Path: path}
	}()

	// Function "Start" - owner of "rec" Chan
	// "Start" is responsible for closing this channel.
	// The channel should be closed if one of two events happens:
	// 1. All tasks are completed "recCount.Wait() - unlocked"
	// 2. The context is canceled "all workers was stoped"

	// 1.
	// We wait for the first event and send a signal about it
	go func() {
		recCount.Wait()

		recClose.L.Lock()
		defer recClose.L.Unlock()

		recClose.Signal()
	}()

	// 2.
	// The second event can occur in the worker pool.
	// Since workers are also senders of data to the "rec" channel,
	// they also send a signal about the completion of their work
	// (when the context is canceled, for example)

	// Here we wait until one of two events happens.
	go func() {
		recClose.L.Lock()
		defer recClose.L.Unlock()

		recClose.Wait()
		close(rec)
	}()

	return rec
}

```

#### "Workhorses" (`worker.Worker` and `worker.RunPool`)

The worker pool is a set of concurrently running goroutines, each executing the task processing logic.

```Go
// Worker defines the interface for a task handler
// (a small overhead for a future project)
type Worker interface {
	run(inp, out, rec chan *queue.Task)
}

// NewWorker creates a new worker instance
func NewWorker() Worker {
	return &fetcher{}
}

// fetcher - implementation of the worker for file system traversal
type fetcher struct {}

// run - the main logic of the worker
func (item *fetcher) run(inp, out, rec chan *queue.Task) {
	for currentTask := range inp { // Read tasks from the input channel
		func() { // Anonymous function for defer
			defer recCount.Done() // Decrement the counter upon completion of the current task

			objects, err := readDir(currentTask.Path) // Read directory contents
			if err != nil {
				fmt.Printf("Objects read error. Path:%s  error:%s\n", currentTask.Path, err.Error())
				return
			}

			for _, object := range objects {
				objectPath := filepath.Join(currentTask.Path, object.Name())

				if object.IsDir() { // If it's a directory
					recCount.Add(1)                       // Increment counter for a new recursive task
					rec <- &queue.Task{Size: 0, Path: objectPath} // Send to the recursive tasks channel
					continue
				}

				// If it's a file
				objectInfo, err := object.Info()
				if err != nil {
					fmt.Printf("Object-info read error. Path:%s  error:%s\n", currentTask.Path, err.Error())
					continue
				}
				out <- &queue.Task{Size: objectInfo.Size(), Path: objectPath} // Send file info to the output channel
			}
		}()
	}
}

// RunPool starts a pool of workers
func RunPool(runWorker Worker, amt int, inp, out, rec chan *queue.Task) {
	var workers sync.WaitGroup // WaitGroup to track the worker goroutines themselves

	for range amt { // Create 'amt' workers
		workers.Add(1)
		go func() {
			defer workers.Done()
			runWorker.run(inp, out, rec)
		}()
	}

	// This goroutine waiting for all workers to complete and closing the main output channel
	go func(currentWorker Worker, outChan chan *queue.Task) {
		workers.Wait()
		close(outChan)

		// 2.
		// For function "Start" (owner of "rec" Chan) we send signal about second event
		// (when workers are complete or the context is canceled and "rec" Chan can be closed)
		recClose.L.Lock()
		defer recClose.L.Unlock()

		recClose.Signal()
	}(runWorker, out)
}
```
`objects, err := readDir(currentTask.Path)` in the code above - custom implementation of `os.ReadDir` without `slices.SortFunc` of entries (if you need, `os.ReadDir` can be used)

```Go
func readDir(name string) ([]os.DirEntry, error) {
	file, err := os.Open(name)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	dirs, err := file.ReadDir(-1)
	return dirs, err
}
```

#### Tying Everything Together (`main.go`)

The `main` function acts as the orchestrator, connecting all components and launching the pipeline.

```Go
package main

import (
	"context"
	"fmt"
	"main/queue"   // Import your queue package
	"main/worker" // Import your worker package
)

func main() {
	// Initialize context to manage the program's lifecycle
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// rec - the channel through which worker.Start will provide initial and "recursive" tasks
	rec := worker.Start(".") // Start traversal from the current directory

	// inp - the channel from which workers will read. It is the output of your queue,
	// which, in turn, reads tasks from the 'rec' channel.
	inp := queue.OutQueue(ctx, queue.InpQueue(rec))

	// out - the final channel for receiving traversal results (file information)
	out := make(chan *queue.Task)

	// Launching the worker pool:
	// NewWorker() - creates a new instance of the task handler
	// 2 - the number of parallel workers
	// inp - input channel for workers
	// out - output channel for traversal results (files)
	// rec - channel for "recursive" tasks (subdirectories)
	worker.RunPool(worker.NewWorker(), 2, inp, out, rec)

	// Main loop: read and print results from the final 'out' channel.
	// The loop will terminate when the 'out' channel is closed.
	for task := range out {
		fmt.Printf("%d %s\n", task.Size, task.Path)
	}

	fmt.Println("Traversal completed.")
}
```

### How It Works: Flow of Execution

1.  `main` calls `worker.Start(".")`, which:
    * Increments `recCount` by 1.
    * Starts a goroutine that sends the root directory (`.`) to the `rec` channel.
    * Starts a goroutine that waits signal then `rec` must closes.
2.  `main` initializes `inp := queue.OutQueue(ctx, queue.InpQueue(rec))`. Your queue begins reading from `rec` and providing tasks via `inp`.
3.  `main` calls `worker.RunPool`, which launches 2 worker goroutines. These workers start reading tasks from `inp`.
4.  When a worker receives a task (directory) from `inp`:
    * It decrements `recCount` (`defer recCount.Done()`).
    * Reads the directory contents.
    * For each **file**: sends its information to the `out` channel (final output).
    * For each **subdirectory**: increments `recCount` by 1 and sends the path to this subdirectory back to the `rec` channel. This new task enters the queue, then `inp`, to be processed by any available worker.
5.  When all directories and subdirectories are processed, `recCount` eventually becomes zero.
6.  The goroutine launched by `worker.Start`, which was waiting for `recCount.Wait()`, receives the signal and **closes the `rec` channel**.
7.  The closing of `rec` signals `queue.InpQueue` to complete, which leads to the closing of the internal queue and then the closing of the `inp` channel (output from the queue).
8.  The closing of `inp` leads to the termination of all workers in the pool (`for range inp` loop finishes).
9.  The goroutine launched by `worker.RunPool`, which was waiting for `workers.Wait()`, receives the signal and **closes the final `out` channel**.
10. `main` stops reading from `out` and exits.

Running the example produces output similar to this:
(this is the content of the current directory :)

```
PS D:\go\go-sample-recursion> go run .
0 .gitignore
26 go.mod
8 .git\COMMIT_EDITMSG
1079 LICENSE.txt
371 .git\config
1361 main.go
73 .git\description
2 README.md
116 .git\FETCH_HEAD
23 .git\HEAD
1561 queue\queue.go
750 .git\index
8301 queue\queue_test.go
41 .git\ORIG_HEAD
1942 worker\worker.go
...
465 .git\logs\refs\remotes\origin\master
41 .git\refs\remotes\origin\master
Traversal completed.
PS D:\go\go-sample-recursion>
```

### Conclusion

The presented approach to parallel traversal of recursive structures in Go demonstrates the language's power and flexibility. By utilizing channels and `sync.WaitGroup` to manage task flow and synchronization, we have created a solution that:
* **Efficiently parallelizes** processing.
* **Eliminates stack depth issues**, allowing the processing of structures with arbitrary nesting levels.
* **Ensures reliable shutdown** of all pipeline components.
* **Scales** through the use of a worker pool.

This pattern is applicable not only to file systems but also to any other hierarchical data, making it a valuable tool in a Go developer's arsenal.

The full source code is available at the link:
[https://github.com/andrey-matveyev/go-sample-recursion](https://github.com/andrey-matveyev/go-sample-recursion)

### P.S. (Postscript)
It's worth noting that in this article, I intentionally did not conduct detailed benchmarks comparing the performance of the presented parallel traversal implementation against Go's standard `filepath.Walk` function. The primary focus of this publication was not to determine which method is faster, but rather to illustrate an architectural pattern for parallelizing recursive processes and mitigating stack overflow issues.

However, based on my personal observations and without formal measurements, `filepath.Walk` feels slower, even when comparing it to our solution running with a single worker (`fetcher`). And yeah, I remember about disk cache. This is an informal observation, and formal benchmarks would be necessary to confirm it.

If any readers are inspired to build such benchmarks and compare the performance metrics, I would be genuinely interested in seeing the results and insights. Feel free to share your findings!