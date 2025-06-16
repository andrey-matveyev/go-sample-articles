# File Duplicate Detector. Go implementation.

### SearchEngine, Processing Pipeline, Usage and Result.

"Truth is not born pure from the earth; it requires refinement from the superfluous to shine in its essence."
— Ancient Wisdom

Previous articles:

- [Part #1. Building a Queue for Go Pipelines](https://dev.to/andrey_matveyev/building-a-queue-for-go-pipelines-24b)
- [Part #2. Parallel Traversal of Recursive Structures in Go.](https://dev.to/andrey_matveyev/parallel-traversal-of-recursive-structures-in-go-12kn)
- [Part #3. Go along the Pipeline: Sizer, Hasher, Matcher.](https://dev.to/andrey_matveyev/go-along-the-pipeline-sizer-hasher-matcher-a2h)

---

In previous parts of our series on the **File Duplicate Detector**, we thoroughly examined individual components: `Workers` that perform specific tasks (file discovery, size determination, hash calculation, byte-by-byte comparison); `Queues` that ensure reliable data transfer between workers; and `Checkers` that optimize the duplicate detection process by preventing redundant work.

Now it's time to put this puzzle together and understand how these independent, yet interconnected parts form a powerful system for detecting duplicate files. The `SearchEngine` component and the data processing pipeline are central to this process.

### 1. `SearchEngine`: The Heart of Orchestration.
The `SearchEngine` is the brain of the entire system. Its main task is to launch and coordinate all stages of the file processing pipeline. It does not perform direct file operations but acts as a conductor, ensuring the correct execution of all operations:

- **Initialization**: Upon launch, the `SearchEngine` initializes internal structures, such as a `sync.WaitGroup` to track the completion of all workers, and `metrics` for collecting statistics.

- **Pipeline Launch**: The `Run` method starts the main processing pipeline in a separate goroutine, allowing it to operate asynchronously.

- **Lifecycle Management**: The `SearchEngine` manages a cancellation context (`context.Context`), allowing for the graceful shutdown of all workers upon receiving a cancellation signal (e.g., `Ctrl+C`).

- **Result Collection**: After the pipeline completes its work, the `SearchEngine` collects and provides the final results via the `GetResult()` method, as well as current progress via `GetProgress()`.

Here's the `SearchEngine` code, with inline comments for clarity::

```Go
// SearchEngine defines the interface for the search engine.
type SearchEngine interface {
    // Run starts the duplicate detection process.
	Run(ctx context.Context, rootPath string, callback func())
    // GetProgress returns the current progress statistics as a JSON byte slice.
	GetProgress() []byte
    // GetResult returns the final duplicate detection result.
	GetResult() *Result
}

// GetEngine returns a new SearchEngine instance.
func GetEngine() SearchEngine {
	return &searchEngine{}
}

// searchEngine is the concrete implementation of SearchEngine.
type searchEngine struct {
	rootPath  string
	callback  func()         // Callback function, called upon completion.
	poolCount sync.WaitGroup // WaitGroup for all worker pools.
	result    *Result        // Final duplicate detection result.
	metrics   *metrics       // Performance and progress metrics.
}

func (item *searchEngine) Run(ctx context.Context, rootPath string, callback func()) {
	item.rootPath = rootPath
	item.callback = callback
	item.metrics = &metrics{}
	item.metrics.StartTime = time.Now()
	go item.runPipeline(ctx)
}

// GetProgress returns the current progress statistics as a JSON byte slice.
func (item *searchEngine) GetProgress() []byte {
	item.metrics.Duration = time.Since(item.metrics.StartTime) // Update duration.
	jsonData, err := json.Marshal(item.metrics)                // Marshal metrics to JSON.
	if err != nil {
		slog.Info("Marshalling error",
			slog.String("method", "json.Marshal"),
			slog.String("error", err.Error()))
	}
	return jsonData // Return JSON data.
}

func (item *searchEngine) GetResult() *Result {
	return item.result
}
```

### 2. Building the Processing Pipeline: pipeline()

The most interesting part of the `SearchEngine` is the `pipeline()` method, which is responsible for constructing the entire data processing pipeline. This demonstrates the principles of pipeline processing and concurrent programming in Go.

The pipeline consists of several sequential stages, each represented by a pool of workers and its own queue. Data (represented by *task) is passed from one stage to the next via Go channels:

- `fileGenerator` (**File Collector**):

  - This is the first stage. It launches `fetcher` workers that traverse the file system, recursively scanning directories and sending tasks (`*task`) for each found file and directory.

  - It's important to note that `fileGenerator` also contains a `dispatcher` that coordinates recursive directory traversal and completion signals for the `fetchers`.

  - The output of `fileGenerator` is a channel that provides tasks containing the path and size of each file.

- `runPool(&sizer{}, N, ...)` (**Size Determiner**):

  - The output channel of `fileGenerator` becomes the input for the `sizer` pool.

  - `sizers` filter files by size (e.g., excluding zero-byte files) and use a `checker` for initial screening of files with unique sizes. If multiple files of the same size are found, they are passed to the next stage.

- `runPool(&hasher{}, N, ...)` (**Hash Calculator**):

  - The output channel of the `sizers` becomes the input for the `hasher` pool.

  - `hashers` calculate the CRC32 hash for a portion of the file. At this stage, a second, more precise, duplicate filtering occurs: files with different hashes are guaranteed not to be duplicates.

  - A `checker` is used here to determine if the hash is already known (a potential duplicate).

- `runPool(&matcher{}, N, ...)` (**Byte-by-Byte Comparator**):

  - The output channel of the `hashers` becomes the input for the `matcher` pool.

  - `matchers` perform the most resource-intensive action: byte-by-byte comparison of files whose size and partial hash have matched. Only at this stage is it definitively confirmed that two files are identical.

  - A `checker` is used to manage groups of potential duplicates to avoid redundant comparisons and track already verified files.

- `...runPipeline(ctx context.Context)` (**Result Generator**):

  - The output channel of the `matchers` (containing only confirmed duplicates) is fed into the `resultQueue`. From this queue, the code in `runPipeline()` executes as the final stage, responsible for collecting and grouping the paths of all confirmed duplicate files, ultimately preparing the data for the final `Result` structure.

```Go
// Launching the pipeline and processing the results
func (item *searchEngine) runPipeline(ctx context.Context) {
	predResult := newPredResult()

	for task := range item.pipeline(ctx) {
		pathList, detected := predResult.List[task.key]
		if detected {
			predResult.List[task.key] = append(pathList, task.info.path)
		} else {
			pathList := make([]string, 1)
			pathList[0] = task.info.path
			predResult.List[task.key] = pathList
		}
	}
	item.result = result(predResult) // Converting predResult to final result
	item.poolCount.Wait()
	item.callback()
}
// Building pipeline
func (item *searchEngine) pipeline(ctx context.Context) chan *task {
	rec := make(chan *task)
	out := fileGenerator(item.rootPath, 4, rec, fetchQueue(ctx, rec, item.metrics), item)
	out = runPool(&sizer{}, 1, sizeQueue(ctx, out, item.metrics), newCheckList(), item)
	out = runPool(&hasher{}, 6, hashQueue(ctx, out, item.metrics), newCheckList(), item)
	out = runPool(&matcher{}, 8, matchQueue(ctx, out, item.metrics), newCheckList(), item)
	out = resultQueue(ctx, out, item.metrics)
	return out
}
```

All these stages are connected by channels, and each `runPool` launches a fixed number (`amt`) of worker goroutines that process incoming tasks in parallel. `metrics` collects statistics for each stage, allowing progress to be monitored.

### 3. Metrics and Monitoring (`metrics.go`, `monitor.go`)

One of the advantages of this implementation is its built-in mechanism for collecting metrics and monitoring progress. This allows not only tracking the process status in real-time but also analyzing the performance of each stage.

The `metrics` and `queueStat` structures (file `metrics.go`) collect data on the number of processed files (`Count`) and their total size (`Size`) at the input (`Inp`) and output (`Out`) of each queue (pipeline stage).

```Go
// metrics.go
type metrics struct {
	StartTime time.Time     `json:"start"`
	Duration  time.Duration `json:"duration"`
	Fetch     queueStat     `json:"fetch"`
	Size      queueStat     `json:"size"`
	Hash      queueStat     `json:"hash"`
	Match     queueStat     `json:"match"`
	Pack      queueStat     `json:"pack"`
}

type queueStat struct {
	Inp statistic `json:"inpQueue"`
	Out statistic `json:"outQueue"`
}
```

The `counter` function (also in `metrics.go`) is a channel wrapper that increments the corresponding metrics as a task passes through the channel.

The `monitor.go` file contains the logic for displaying progress. In the `progressMonitor` function, metric data is periodically retrieved via engine.`GetProgress()` (which returns a JSON representation of metrics) and printed to the console.

```Go
// monitor.go
func progressMonitor(ctx context.Context, engine fdd.SearchEngine) {
	// ...
	for {
		err := json.Unmarshal(engine.GetProgress(), &stat)
		// ...
		fmt.Println(
			runtime.NumGoroutine(),
			"folder",
			metrics(stat.Fetch),
			"fetch",
			// ...
			stat.Duration,
		)
		time.Sleep(10 * time.Second)
	}
}
```

This allows real-time viewing of:

- The number of active goroutines.

- Processing progress at each stage (number of files and their total size).

- Total running duration.

- Filtering performance at the sizer, hasher, and matcher stages – how many files were "discarded" at each stage.


### 4. Example Usage
Using the `File Duplicate Detector` from `main.go` is quite straightforward:

```Go
func main() {
	runtime.GOMAXPROCS(runtime.NumCPU())
	
    // Creating config application
	config := newConfig()

    // Saving default logger
	oldLogger := slog.Default()

    // Creating newLogger and set as default
	logFile := newLogFile(config.LoggerFileName)
	newLogger(
		withLevel(config.LoggerLevel),
		withAddSource(config.LoggerAddSource),
		withLogFile(logFile),
		withSetDefault(true),
	)

	// Creating context with Cancel
	ctx := context.Background()
	ctx, cancel := context.WithCancel(ctx)

	// Preparing cancel-mechanism (over signal <Ctrl+C>)
	ch := make(chan os.Signal, 1)
	signal.Notify(ch, os.Interrupt)
	go func() {
		<-ch
		fmt.Println("Canselling... ")
		fmt.Println("The application needs to close all resources and save the current result.")
		fmt.Println("Please wait...")
		cancel()
	}()

	engine := fdd.GetEngine()
	// Preparing callback function (event about all tasks completed)
	var wg sync.WaitGroup
	callback := func() {
		wg.Done()
	}

	// Running main work and progress-monitor
	wg.Add(1)
	engine.Run(ctx, config.RootPath, callback)
	go progressMonitor(ctx, engine)
	wg.Wait()

	// Saving result to file
	saveResult(engine, config.ResultFileName)

	// Saving logs
	logFile.Close()

	// Returning default logger
	slog.SetDefault(oldLogger)

	// Printing total statistic
	printStatistic(engine, config)
}
```

As seen in the example, you:

- Create a `context.Context` with cancellation capability.

- Set up OS signal handling for graceful termination.

- Obtain a `SearchEngine` instance.

- Call `Run()` with the root path for scanning and a `callback` function that will be invoked upon completion.

- Launch `progressMonitor` to display progress.

- Wait for all operations to complete using a `sync.WaitGroup`.

- Retrieve and save (GetResult()) the found duplicates.

### 5. **Obtained Results** (`Result`)
The final result of the `SearchEngine`'s operation is available via the `GetResult()` method and is returned as a `Result` structure:

```Go
type Result struct {
	List []element
}

type element struct {
	header
	body
}

type header struct {
	Size  int64
	Hash  uint32
	Group int
}

type body struct {
	Paths []string
}
```

#### Example of application execution from the console (scanning an SSD disk C:):

```
PS D:\go\go-sample-detector> go run .
---- CURRENT CONFIGURATION ----
Root Path:             c:\
File name for results:     fdd-result.txt
File name for logs:        fdd-output.log
Logging level (info/debug): debug
Adds source info in logs:  false
-------------------------------
Progress (every 10 seconds):
45 folder 1_0_1 fetch 0_0_0 size 0_0_0 hash 0_0_0 match 0_0_0 result 0s
50 folder 10849_6798_4051 fetch 41712_0_41712 size 29731_20019_9712 hash 1654_757_897 match 432_0_432 result 10.0514188s
...
50 folder 827791_5379_822412 fetch 2175225_1_2175224 size 2136075_1166018_970057 hash 737102_171_736931 match 725972_0_725972 result 11m30.1975563s
50 folder 838528_3506_835022 fetch 2249794_0_2249794 size 2208711_1225693_983018 hash 750085_965_749120 match 737973_0_737973 result 11m40.1993708s
31 folder 843222_0_843222 fetch 2270482_0_2270482 size 2229228_1234927_994301 hash 761939_5_761934 match 750026_6_750020 result 11m50.2018749s
31 folder 843222_0_843222 fetch 2270482_0_2270482 size 2229228_1219410_1009818 hash 773913_0_773913 match 761518_0_761518 result 12m0.2085391s
31 folder 843222_0_843222 fetch 2270482_0_2270482 size 2229228_1205527_1023701 hash 787439_42_787397 match 774824_0_774824 result 12m10.2143553s
...
31 folder 843222_0_843222 fetch 2270482_0_2270482 size 2229228_24236_2204992 hash 1722277_1662_1720615 match 1700204_1_1700203 result 22m0.3185755s
5 folder 843222_0_843222 fetch 2270482_0_2270482 size 2229228_0_2229228 hash 1735310_0_1735310 match 1714249_0_1714249 result 22m10.3194308s
4 folder 843222_0_843222 fetch 2270482_0_2270482 size 2229228_0_2229228 hash 1735310_0_1735310 match 1714249_0_1714249 result 22m20.320499s
4 folder 843222_0_843222 fetch 2270482_0_2270482 size 2229228_0_2229228 hash 1735310_0_1735310 match 1714249_0_1714249 result 22m30.3211656s
------- TOTAL STATISTIC -------
Time of start:       11:29:09
Time of ended:       11:51:41
Duration:        22m31.7249332s
Total processed <count (size Mb)>:
- folders:           843222
- files:             2270482 (178753.517 Mb)
Performance of filtration <inp-filtered-out (out %)>:
- sizer:             2270482     41254       2229228 (98.18 %)
- hasher:            2229228     493918      1735310 (77.84 %)
- matcher:           1735310     21061       1714249 (98.79 %)
Found duplicates <count (size Mb)>:
- groups of files:   311941
- files:             1714249 (63811.304 Mb)
File with result: fdd-result.txt
File with logs: fdd-output.log
-------------------------------
```

#### HDD Disk Scan Example:

```
PS D:\go\go-sample-detector> go run .
---- CURRENT CONFIGURATION ----
Root Path:             d:\
File name for results:     fdd-result.txt
File name for logs:        fdd-output.log
Logging level (info/debug): debug
Adds source info in logs:  false
-------------------------------
Progress (every 10 seconds):
17 folder 0_0_0 fetch 0_0_0 size 0_0_0 hash 0_0_0 match 0_0_0 result 520.5µs
50 folder 382_224_158 fetch 1910_0_1910 size 606_279_327 hash 277_155_122 match 108_0_108 result 10.0512721s
...
50 folder 6712_454_6258 fetch 54626_0_54626 size 32410_25710_6700 hash 3972_2552_1420 match 1334_0_1334 result 5m20.0859867s
50 folder 6952_73_6879 fetch 55794_1_55793 size 33647_26147_7500 hash 4658_3204_1454 match 1365_0_1365 result 5m30.0865653s
31 folder 7008_0_7008 fetch 56138_0_56138 size 34005_26153_7852 hash 4791_3313_1478 match 1389_0_1389 result 5m40.0886867s
31 folder 7008_0_7008 fetch 56138_0_56138 size 34005_25868_8137 hash 4922_3373_1549 match 1451_0_1451 result 5m50.0892709s
...
31 folder 7008_0_7008 fetch 56138_0_56138 size 34005_567_33438 hash 22755_20341_2414 match 2291_0_2291 result 12m30.1476568s
31 folder 7008_0_7008 fetch 56138_0_56138 size 34005_315_33690 hash 22961_20490_2471 match 2353_0_2353 result 12m40.1494179s
20 folder 7008_0_7008 fetch 56138_0_56138 size 34005_0_34005 hash 23323_20744_2579 match 2459_0_2459 result 12m50.1500104s
20 folder 7008_0_7008 fetch 56138_0_56138 size 34005_0_34005 hash 23323_20608_2715 match 2591_0_2591 result 13m0.1504921s
...
20 folder 7008_0_7008 fetch 56138_0_56138 size 34005_0_34005 hash 23323_423_22900 match 22122_0_22122 result 29m10.2776533s
12 folder 7008_0_7008 fetch 56138_0_56138 size 34005_0_34005 hash 23323_0_23323 match 22555_0_22555 result 29m20.2791754s
11 folder 7008_0_7008 fetch 56138_0_56138 size 34005_0_34005 hash 23323_0_23323 match 22557_0_22557 result 29m30.2803337s
11 folder 7008_0_7008 fetch 56138_0_56138 size 34005_0_34005 hash 23323_0_23323 match 22557_0_22557 result 29m40.2814158s
------- TOTAL STATISTIC -------
Time of start:       12:03:35
Time of ended:       12:33:22
Duration:        29m46.4419118s
Total processed <count (size Mb)>:
- folders:             7008
- files:               56138 (645181.041 Mb)
Performance of filtration <inp-filtered-out (out %)>:
- sizer:               56138     22133       34005 (60.57 %)
- hasher:              34005     10682       23323 (68.59 %)
- matcher:             23323       764       22559 (96.72 %)
Found duplicates <count (size Mb)>:
- groups of files:     7243
- files:               22559 (15674.642 Mb)
File with result: fdd-result.txt
File with logs: fdd-output.log
-------------------------------
```

#### Example Logging Output (from fdd-output.log)

```Log
time=2025-06-15T11:29:09.260+03:00 level=DEBUG msg="InpProcess of Queue - started." poolName=fetchers
time=2025-06-15T11:29:09.260+03:00 level=DEBUG msg="OutProcess of Queue - started." poolName=fetchers
time=2025-06-15T11:29:09.260+03:00 level=DEBUG msg="Worker-pool - started." workerType=fdd.fetcher
time=2025-06-15T11:29:09.306+03:00 level=DEBUG msg="Worker-pool - started." workerType=*fdd.sizer
time=2025-06-15T11:29:09.306+03:00 level=DEBUG msg="Worker-pool - started." workerType=*fdd.hasher
time=2025-06-15T11:29:09.306+03:00 level=DEBUG msg="Worker-pool - started." workerType=*fdd.matcher
time=2025-06-15T11:29:09.308+03:00 level=DEBUG msg="InpProcess of Queue - started." poolName=sizers
time=2025-06-15T11:29:09.308+03:00 level=DEBUG msg="OutProcess of Queue - started." poolName=sizers
time=2025-06-15T11:29:09.308+03:00 level=DEBUG msg="InpProcess of Queue - started." poolName=matchers
time=2025-06-15T11:29:09.308+03:00 level=DEBUG msg="InpProcess of Queue - started." poolName=hashers
time=2025-06-15T11:29:09.308+03:00 level=DEBUG msg="OutProcess of Queue - started." poolName=hashers
time=2025-06-15T11:29:09.308+03:00 level=DEBUG msg="OutProcess of Queue - started." poolName=matchers
time=2025-06-15T11:29:09.308+03:00 level=DEBUG msg="InpProcess of Queue - started." poolName=packer
time=2025-06-15T11:29:09.308+03:00 level=DEBUG msg="OutProcess of Queue - started." poolName=packer
time=2025-06-15T11:29:09.311+03:00 level=INFO msg="Objects read error." item=*fdd.fetcher method=readDir() error="open c:\\$Recycle.Bin\\S-1-5-18: Access is denied." path=c:\$Recycle.Bin\S-1-5-18
...
time=2025-06-15T11:39:52.958+03:00 level=INFO msg="File open error." item=*fdd.hasher method=os.Open() error="open c:\\Windows\\System32\\restore\\MachineGuid.txt: Access is denied." path=c:\Windows\System32\restore\MachineGuid.txt
time=2025-06-15T11:39:57.011+03:00 level=INFO msg="File open error." item=*fdd.hasher method=os.Open() error="open c:\\Windows\\System32\\wbem\\AutoRecover\\3FFDD473F026FB198DA9FA65EE71383C.mof: Access is denied." path=c:\Windows\System32\wbem\AutoRecover\3FFDD473F026FB198DA9FA65EE71383C.mof
time=2025-06-15T11:40:55.247+03:00 level=DEBUG msg="InpProcess of Queue - stoped." poolName=fetchers
time=2025-06-15T11:40:55.247+03:00 level=DEBUG msg="OutProcess of Queue - stopped because queue is done and empty." poolName=fetchers
time=2025-06-15T11:40:55.250+03:00 level=DEBUG msg="Worker-pool - stoped." workerType=fdd.fetcher
time=2025-06-15T11:40:55.250+03:00 level=DEBUG msg="InpProcess of Queue - stoped." poolName=sizers
time=2025-06-15T11:40:55.252+03:00 level=DEBUG msg="OutProcess of Queue - stopped because queue is done and empty." poolName=sizers
time=2025-06-15T11:40:55.252+03:00 level=DEBUG msg="Worker-pool - stoped." workerType=*fdd.sizer
time=2025-06-15T11:40:55.252+03:00 level=DEBUG msg="InpProcess of Queue - stoped." poolName=hashers
...
time=2025-06-15T11:50:53.101+03:00 level=INFO msg="File open error." item=*fdd.hasher method=os.Open() error="open c:\\Windows\\System32\\wbem\\AutoRecover\\DA736886F13A0E2EE2265319FB376753.mof: Access is denied." path=c:\Windows\System32\wbem\AutoRecover\DA736886F13A0E2EE2265319FB376753.mof
time=2025-06-15T11:51:19.521+03:00 level=DEBUG msg="OutProcess of Queue - stopped because queue is done and empty." poolName=hashers
time=2025-06-15T11:51:19.522+03:00 level=DEBUG msg="Worker-pool - stoped." workerType=*fdd.hasher
time=2025-06-15T11:51:19.522+03:00 level=DEBUG msg="InpProcess of Queue - stoped." poolName=matchers
time=2025-06-15T11:51:19.522+03:00 level=DEBUG msg="OutProcess of Queue - stopped because queue is done and empty." poolName=matchers
time=2025-06-15T11:51:19.522+03:00 level=DEBUG msg="Worker-pool - stoped." workerType=*fdd.matcher
time=2025-06-15T11:51:19.522+03:00 level=DEBUG msg="InpProcess of Queue - stoped." poolName=packer
time=2025-06-15T11:51:19.522+03:00 level=DEBUG msg="OutProcess of Queue - started." poolName=packer
```
DEBUG level logs show the start and stop of each worker pool (`Worker-pool - started./stoped.`) and queue processing (`InpProcess/OutProcess of Queue - started./stoped.`). INFO level logs often indicate file or folder access errors, for example, to system directories (`Access is denied.`). This is expected behavior, as the application attempts to access all files in the specified root directory.

#### Example Result File (fdd-result.txt)

After completion, the application saves the results to a text file specified in the configuration (`fdd-result.txt`). The output format groups duplicates by size, hash, and group ID, then lists the paths to the duplicate files.

```
      2  {32  3876609034  0}
d:\HP_Drivers_for_Win10\SWSetup\SP92183\Graphics\ocl_cpu_version.ini
d:\HP_Drivers_for_Win10\SWSetup\SP95347\Graphics\ocl_cpu_version.ini
      2  {33  554973275  0}
d:\HP_Drivers_for_Win10\SWSetup\SP92183\DisplayAudio\6.16\version.ini
d:\HP_Drivers_for_Win10\SWSetup\SP95347\DisplayAudio\6.16\version.ini
      3  {33  4084763797  0}
d:\HP_Drivers_for_Win10\SWSetup\SP57014\Driver1\silentsetup.bat
d:\HP_Drivers_for_Win10\SWSetup\SP57014\Driver2\silentsetup.bat
d:\HP_Drivers_for_Win10\SWSetup\SP57014\silentsetup.bat
      2  {41  388051727  0}
d:\go\go-sample-queue\.git\refs\heads\master
d:\go\go-sample-queue\.git\refs\remotes\origin\master
      2  {41  954715591  0}
d:\go\go-sample-detector\.git\ORIG_HEAD
d:\go\go-sample-detector\.git\refs\heads\master
      3  {41  1012172877  0}
d:\go\go-sample-recursion\.git\ORIG_HEAD
d:\go\go-sample-recursion\.git\refs\heads\master
d:\go\go-sample-recursion\.git\refs\remotes\origin\master
```

Each group of duplicates starts with a line containing:

- The number of files in the group (e.g., 2 or 3).

- The structure {size hash group}:

  - size: File size in bytes.

  - hash: CRC32 hash of the files.

  - group: Group ID (in this case, 0). This field is used for grouping identical files that have the same `size` and `hash`. If files have the same `size` and `hash` but their content differs (which is determined at the `matcher` stage), they will be assigned a different `Group` ID, and they will not be included in the same group of duplicates in the final result.

- Followed by the full paths to each duplicate file in that group.

### 6. Conclusion
The presented implementation of duplicate file detection in Golang demonstrates the power of the pipelined approach and parallel processing. The use of channels for data transfer between stages, worker pools for parallel execution of tasks, and a well-designed checker mechanism makes the solution efficient and scalable. Built-in metrics and monitoring significantly simplify debugging and performance analysis.

I was surprised by the performance, despite the fact that I almost didn't think about optimization. On the contrary - the overhead in the form of monitoring, which I used to observe the load in real time - slows down the work.

Of course, there is always room for improvement. Potential enhancements could include:

- Support for very large files using streaming processing and more sophisticated hashing algorithms.

- Or more thoughtful use of disk cache (for example, not allowing the cache to "cool down" after the hasher before the matcher).

Nevertheless, this solution serves as a starting point for understanding and implementing pipelined systems in Golang.

The full source code is available at the link:
[https://github.com/andrey-matveyev/go-sample-detector](https://github.com/andrey-matveyev/go-sample-detector)