# Building a Queue for Go Pipelines

> Hello everyone! This is my first post on this platform, and I'm kicking it off with a small publication. Happy reading!

---
In the Go world, pipelines built on channels are powerful tools for organizing streaming data processing. However, standard channels don't always provide all the necessary flexibility, especially when you need to manage data flow between producers and consumers operating at different speeds, and avoid busy-waiting.
In this article, we'll explore how to build such a queue in Go, leveraging its native concurrency primitives to create a non-blocking, efficient, and context-aware solution.

### The Challenge: Rate Mismatches in Pipelines

Consider a typical data processing pipeline:
`Producer (Fast) -> Stage 1 (Medium) -> Stage 2 (Slow) -> Consumer (Variable)`
If Stage 1 is much faster than Stage 2, data can pile up, potentially leading to memory issues (or other resources) or blocking the entire pipeline. Conversely, if Stage 1 is slow and Stage 2 is fast, Stage 2 might spend too much time waiting for data (busy-waiting). A queue acts as a buffer, decoupling these stages and allowing them to operate at their own pace.

## Designing Our Queue

Our Go queue will have the following characteristics:
1. **Non-blocking Producer/Consumer:** Adding to or taking from the queue should not block the respective goroutines unnecessarily.
2. **Context-aware Consumer:** The consumer (reading from the queue) should respect context.Context for cancellation, allowing graceful shutdown.
3. **Completion Signaling:** The queue should clearly signal when no more items will arrive from the producer, allowing the consumer to finish processing and shut down cleanly.
4. **Simplicity & Flexibility & Efficiency:** Leveraging Go's built-in container/list for the underlying data structure and channels for synchronization.

Let's imagine our queue as a friendly middleman, taking tasks from a fast sender and passing them to a slower receiver, ensuring everyone is happy and the process flows smoothly. You can see the visualization of the idea in the main picture of the publication.

## The Code Implementation

`queue` struct:

- 
`mtx sync.Mutex`: Protects `queueTasks` from concurrent access, ensuring thread safety for `push` and `pop`.

- 
`innerChan chan struct{}`: A buffered channel used for signaling. `inpProcess` sends a signal to `outProcess` whenever a new task is added. The buffer of 1 prevents `inpProcess` from blocking if `outProcess` isn't immediately ready to consume the signal. When `inpProcess` finishes, it closes `innerChan` to notify `outProcess` of completion.

- 
`queueTasks *list.List`: Go's standard library doubly linked list. It's a good fit for a `queue` as `PushBack` and `Remove(Front())` are efficient.

```Go
type queue struct {
	mtx        sync.Mutex
	innerChan  chan struct{}
	queueTasks *list.List
}

func newQueue() *queue {
	item := queue{}
	item.innerChan = make(chan struct{}, 1)
	item.queueTasks = list.New()
	return &item
}

func (item *queue) push(task *Task) {
	item.mtx.Lock()
	item.queueTasks.PushBack(task)
	item.mtx.Unlock()
}

func (item *queue) pop() *Task {
	item.mtx.Lock()
	defer item.mtx.Unlock()

	if item.queueTasks.Len() == 0 {
		return nil
	}
	elem := item.queueTasks.Front()
	item.queueTasks.Remove(elem)
	return elem.Value.(*Task)
}

type Task struct {
	ID   int
	Data string
}
```

### Key Components:

There are two main functions: `InpQueue` (for the producer side) and `OutQueue` (for the consumer side).

#### `inpProcess` Goroutine (Producer Side):

- 
Reads tasks from its input channel (`inp`).

- 
`queue.push(value)`: Adds tasks to the internal `queueTasks` list, protected by `mtx`.

- 
`select { case queue.innerChan <- struct{}{}: default: }`: This is a non-blocking send. It attempts to send a signal to `innerChan`. If `innerChan`'s buffer is full (meaning `outProcess` already knows there are tasks to process and hasn't read the previous signal yet), the default case is taken, and `inpProcess` continues without blocking.

- 
`close(queue.innerChan)`: Crucially, when the `inp` channel is closed (meaning the producer has no more tasks), `inpProcess` closes `innerChan`. This acts as the final signal to `outProcess` that the stream of incoming tasks has ended.

```Go
func InpQueue(inp chan *Task) *queue {
	queue := newQueue()
	go inpProcess(inp, queue)
	return queue
}

func inpProcess(inp chan *Task, queue *queue) {
	for value := range inp {
		queue.push(value)

		select {
		case queue.innerChan <- struct{}{}:
		default:
		}
	}
	close(queue.innerChan)
}
```

#### `outProcess` Goroutine (Consumer Side):

- 
`defer close(out)`: Ensures the output channel `out` is closed when `outProcess` exits, signaling to its downstream consumer that no more data will arrive.

- 
`select { case <-ctx.Done(): ... case _, ok := <-queue.innerChan: ... }`: This `select` statement is the heart of the consumer's behavior:

 - 
`<-ctx.Done()`: Checking the context for cancellation. If the main context is cancelled, outProcess immediately returns, stopping consumption gracefully.

 - 
`_, ok := <-queue.innerChan`: Waits for a signal that new tasks are available or for `innerChan` to be closed. The `ok` boolean indicates whether the channel was closed (`false`) or a value was received (`true`).

- 
**Drain Loop** (`for { task := queue.pop() ... }`): After receiving a signal (or detecting that `innerChan` has been closed), `outProcess` enters a loop to pop all currently available tasks from `queueTasks`. This is important because a single signal might correspond to multiple tasks having been pushed (e.g., if the producer was very fast).

- 
`if !ok { return }`: After exhausting the internal queue, if `innerChan` was found to be closed (`ok` is `false`), it means `inpProcess` has finished, and no more tasks will ever arrive. `outProcess` then safely returns.

```Go
func OutQueue(ctx context.Context, queue *queue) chan *Task {
	out := make(chan *Task)
	go outProcess(ctx, queue, out)
	return out
}

func outProcess(ctx context.Context, queue *queue, out chan *Task) {
	defer close(out)
	for {
		select {
		case <-ctx.Done():
			return
		case _, ok := <-queue.innerChan:
			for {
				task := queue.pop()
				if task != nil {
					select {
					case out <- task:
					case <-ctx.Done():
						return
					}
				} else {
					break
				}
			}
			if !ok {
				return
			}
		}
	}
}
```

## Example Usage

Here's how you might integrate and use this queue in a main function to simulate a producer and a slower consumer:

```Go
func main() {
	startTime := time.Now()
	// Initialize context to manage the entire pipeline
	mainCtx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// 1. Create an initial input channel for tasks
	// In a real system, we get it from the previous pipeline element.
	inpChan := make(chan *queue.Task)

	// 2. Embed our queue into the pipeline:
	// inpChan -> inpQueue (transforms channel to queue) -> outQueue (transforms queue to channel) -> outChan
	// This stage simulates some processing and includes the queue.
	outChan := queue.OutQueue(mainCtx, queue.InpQueue(inpChan))

	// 3. Start a producer goroutine:
	// It will generate tasks and send them to inpChan.
	produced := 0
	go func() {
		fmt.Printf("Producer: started. (%dms)\n", time.Since(startTime).Milliseconds())
		for i := range 5 {
			task := &queue.Task{ID: i, Data: fmt.Sprintf("Task #%d", i)}
			fmt.Printf("Producer: Sending %s  (%dms)\n", task.Data, time.Since(startTime).Milliseconds())
			inpChan <- task
			produced++
			time.Sleep(200 * time.Millisecond) // Simulate producer work
		}
		close(inpChan) // Important: close the input channel when all tasks are sent
		fmt.Printf("Producer: All tasks sent, input channel closed. (%dms)\n", time.Since(startTime).Milliseconds())
	}()

	// 4. Start a consumer goroutine:
	// It will read tasks from outChan (the output of our queue).
	consumed := 0
	go func() {
		fmt.Printf("Consumer: started. (%dms)\n", time.Since(startTime).Milliseconds())
		for task := range outChan {
			consumed++
			fmt.Printf("Consumer: Received %s  (%dms)\n", task.Data, time.Since(startTime).Milliseconds())
			time.Sleep(400 * time.Millisecond) // Simulate a slower consumer
		}
		fmt.Printf("Consumer: All tasks processed, output channel closed. (%dms)\n", time.Since(startTime).Milliseconds())
	}()
	// The pipeline will finish when inpChan closes -> inpProcess finishes ->
	// queue.innerChan closes -> outProcess finishes -> outChan closes.

	/*
	    // Uncomment this code to see how context manages the operation's lifecycle.
	   	time.Sleep(1 * time.Second) // Timeout in case of hang
	   	fmt.Printf("Main: Timeout reached, cancelling context. (%dms)\n", time.Since(startTime).Milliseconds())
	   	cancel()
	*/
	// Small delay for all goroutines to finish after cancellation/completion.
	time.Sleep(3 * time.Second)
	fmt.Printf("-produced: %d tasks, -consumed: %d tasks.\n", produced, consumed)
	fmt.Printf("Main: Application finished. (%dms)\n", time.Since(startTime).Milliseconds())
}
```

Running the example produces output similar to this:

```
PS D:\go\queue> go run .
Producer: started. (0ms)
Producer: Sending Task #0  (0ms)
Consumer: started. (0ms)
Consumer: Received Task #0  (1ms)
Producer: Sending Task #1  (201ms)
Producer: Sending Task #2  (401ms)
Consumer: Received Task #1  (401ms)
Producer: Sending Task #3  (602ms)
Consumer: Received Task #2  (802ms)
Producer: Sending Task #4  (803ms)
Producer: All tasks sent, input channel closed. (1004ms)
Consumer: Received Task #3  (1203ms)
Consumer: Received Task #4  (1603ms)
Consumer: All tasks processed, output channel closed. (2004ms)
-produced: 5 tasks, -consumed: 5 tasks.
Main: Application finished. (3001ms)
```
Notice that Producer is not blocked and works successfully even if Consumer has not yet started.
`Task #0` is produced at `0ms` and consumed at `1ms`. But `Task #1` is produced at `201ms` and consumed at `401ms` — the queue is buffering. The producer finishes sending all tasks by `803ms`, but the consumer continues processing until `1603ms`, demonstrating the decoupling effect of the queue.

How context manages the operation's lifecycle (example):

```
PS D:\go\queue> go run .
Producer: started. (0ms)
Producer: Sending Task #0  (0ms)
Consumer: started. (0ms)
Consumer: Received Task #0  (1ms)
Producer: Sending Task #1  (201ms)
Producer: Sending Task #2  (403ms)
Consumer: Received Task #1  (403ms)
Producer: Sending Task #3  (603ms)
Consumer: Received Task #2  (804ms)
Producer: Sending Task #4  (804ms)
Main: Timeout reached, cancelling context. (1000ms)
Producer: All tasks sent, input channel closed. (1005ms)
Consumer: All tasks processed, output channel closed. (1204ms)
-produced: 5 tasks, -consumed: 3 tasks.
Main: Application finished. (4001ms)
```

## Conclusion
By implementing this queue pattern using Go's sync.Mutex, container/list, and especially channels for inter-goroutine signaling and context.Context for cancellation, we've created a flexible and efficient buffering mechanism for pipelines. This approach allows different stages of your concurrent applications to operate independently, smoothing out variable processing rates and ensuring robust, graceful shutdowns.

The full source code (with testing more cases) is available at the link:
[https://github.com/andrey-matveyev/go-sample-queue](https://github.com/andrey-matveyev/go-sample-queue)

