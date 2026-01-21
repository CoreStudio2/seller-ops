### tf.backend

Source: https://js.tensorflow.org/api/latest/index

Gets the current backend. If no backends have been initialized, this will attempt to initialize the best backend.

```APIDOC
## tf.backend ()

### Description
Gets the current backend. If no backends have been initialized, this will attempt to initialize the best backend. Will throw an error if the highest priority backend has async initialization, in which case you should call 'await tf.ready()' before running other code.

### Method
N/A (Backend function)

### Endpoint
N/A

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
None

### Request Example
```javascript
try {
  const currentBackend = tf.backend();
  console.log(`Current backend: ${currentBackend.binding().name}`);
} catch (e) {
  console.error('Backend initialization failed or is asynchronous.');
}
```

### Response
#### Success Response (N/A)
- **KernelBackend** - An instance of the current backend.

#### Response Example
```json
// Example: An object representing the WebGL backend
```
```

--------------------------------

### tf.layers.convLstm2d Example - JavaScript

Source: https://js.tensorflow.org/api/latest/index

This example demonstrates how to create and apply a tf.layers.convLstm2d layer in TensorFlow.js. It initializes the layer with specified filters and kernel size, then applies it to a sample input tensor.

```javascript
const filters = 3;
const kernelSize = 3;

const batchSize = 4;
const sequenceLength = 2;
const size = 5;
const channels = 3;

const inputShape = [batchSize, sequenceLength, size, size, channels];
const input = tf.ones(inputShape);

const layer = tf.layers.convLstm2d({filters, kernelSize});

const output = layer.apply(input);
```

--------------------------------

### List, Remove, and Move Models

Source: https://js.tensorflow.org/api/latest/index

This section covers listing, removing, and moving models using TensorFlow.js I/O functions. It includes examples for saving a model, listing available models, removing a model, and moving a model between storage mediums.

```APIDOC
## tf.io.listModels

### Description
Lists all the models that have been saved to the browser's local storage or IndexedDB.

### Method
`GET`

### Endpoint
`/tfjs/io/listModels`

### Parameters
None

### Response
#### Success Response (200)
- **Promise<{[url: string]: ModelArtifactsInfo]}>** - A promise that resolves to an object where keys are model URLs and values are model artifact information.

#### Response Example
```json
{
  "localstorage://demo/management/model1": {
    "modelTopology": {},
    "dateSaved": "2023-01-01T12:00:00.000Z",
    " وقد-Saved": "2023-01-01T12:00:00.000Z"
  }
}
```

## tf.io.removeModel

### Description
Removes a model specified by URL from a registered storage medium.

### Method
`DELETE`

### Endpoint
`/tfjs/io/removeModel`

### Parameters
#### Query Parameters
- **url** (string) - Required - A URL to a stored model, with a scheme prefix, e.g., 'localstorage://my-model-1', 'indexeddb://my/model/2'.

### Response
#### Success Response (200)
- **Promise<ModelArtifactsInfo>** - A promise that resolves to the model artifacts information of the removed model.

#### Response Example
```json
{
  "modelTopology": {},
  "dateSaved": "2023-01-01T12:00:00.000Z",
  " وقد-Saved": "2023-01-01T12:00:00.000Z"
}
```

## tf.io.moveModel

### Description
Moves a model from one URL to another. This function supports moving within a storage medium or between different storage mediums.

### Method
`POST`

### Endpoint
`/tfjs/io/moveModel`

### Parameters
#### Request Body
- **sourceURL** (string) - Required - Source URL of the model to move.
- **destURL** (string) - Required - Destination URL for the moved model.

### Request Example
```json
{
  "sourceURL": "localstorage://demo/management/model1",
  "destURL": "indexeddb://demo/management/model1"
}
```

### Response
#### Success Response (200)
- **Promise<ModelArtifactsInfo>** - A promise that resolves to the model artifacts information of the moved model.

#### Response Example
```json
{
  "modelTopology": {},
  "dateSaved": "2023-01-01T12:00:00.000Z",
  " وقد-Saved": "2023-01-01T12:00:00.000Z"
}
```
```

--------------------------------

### Wait for Backend Initialization with tf.ready

Source: https://js.tensorflow.org/api/latest/index

Returns a Promise that resolves when the currently selected or highest-priority backend has finished initializing. Use `await tf.ready()` before executing code that depends on backend initialization, especially for backends with asynchronous setup.

```javascript
async function initializeAndRun() {
  await tf.ready();
  console.log('Backend is ready.');
  // Proceed with TensorFlow.js operations
}
initializeAndRun();
```

--------------------------------

### Mapping a Dataset Transformation in JavaScript

Source: https://js.tensorflow.org/api/latest/index

Maps this dataset through a 1-to-1 transform. This example squares each element in the dataset.

```javascript
const a = tf.data.array([1, 2, 3]).map(x => x*x);
await a.forEachAsync(e => console.log(e));
```

--------------------------------

### Compile and Evaluate a Sequential Model in JavaScript

Source: https://js.tensorflow.org/api/latest/index

This example shows how to create a simple sequential TensorFlow.js model, compile it with a specified optimizer and loss function, and then evaluate its performance on test data. The model.evaluate() method returns the loss value and metrics, which are then printed to the console. This is a standard workflow for assessing model accuracy.

```javascript
const model = tf.sequential({
   layers: [tf.layers.dense({units: 1, inputShape: [10]})]
});
model.compile({optimizer: 'sgd', loss: 'meanSquaredError'});
const result = model.evaluate(
     tf.ones([8, 10]), tf.ones([8, 1]), {batchSize: 4});
result.print();

```

--------------------------------

### Create and Summarize a Multi-Input Model in JavaScript

Source: https://js.tensorflow.org/api/latest/index

This example demonstrates how to create a TensorFlow.js model with multiple inputs using tf.input and tf.model. It then calls the model.summary() method to print a text summary of the model's layers, including their shapes, parameters, and connectivity. This is useful for debugging and understanding model architecture.

```javascript
const input1 = tf.input({shape: [10]});
const input2 = tf.input({shape: [20]});
const dense1 = tf.layers.dense({units: 4}).apply(input1);
const dense2 = tf.layers.dense({units: 8}).apply(input2);
const concat = tf.layers.concatenate().apply([dense1, dense2]);
const output = 
     tf.layers.dense({units: 3, activation: 'softmax'}).apply(concat);

const model = tf.model({inputs: [input1, input2], outputs: output});
model.summary();

```

--------------------------------

### Create Tensor from WebGPU Buffer Data in JavaScript

Source: https://js.tensorflow.org/api/latest/index

Creates a tensor directly from WebGPU buffer data, enabling efficient GPU-to-GPU data transfer. This example shows how to create a GPU buffer from provided data, transfer it to a read-accessible buffer, and then use it to create a TensorFlow.js tensor. It highlights the `zeroCopy` option for potential performance gains, explaining its implications for buffer management.

```javascript
// Pass a `WebGPUData` object and specify a shape yourself.

// This makes it possible for TF.js applications to avoid GPU / CPU sync.
// For example, if your application includes a preprocessing step on the GPU,
// you could upload the GPU output directly to TF.js, rather than first
// downloading the values. Unlike WebGL, this optionally supports zero copy
// by WebGPUData.zeroCopy. When zeroCopy is false or undefined(default), this
// passing GPUBuffer can be destroyed after tensor is created. When zeroCopy
// is true, this GPUBuffer is bound directly by the tensor, so do not destroy
// this GPUBuffer until all access is done.

// Example for WebGPU:
function createGPUBufferFromData(device, data, dtype) {
   const bytesPerElement = 4;
   const sizeInBytes = data.length * bytesPerElement;

   const gpuWriteBuffer = device.createBuffer({
     mappedAtCreation: true,
     size: sizeInBytes,
     usage: GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC
   });
   const arrayBuffer = gpuWriteBuffer.getMappedRange();
   if (dtype === 'float32') {
     new Float32Array(arrayBuffer).set(data);
   } else if (dtype === 'int32') {
     new Int32Array(arrayBuffer).set(data);
   } else {
     throw new Error(
         `Creating tensor from GPUBuffer only supports` +
         `'float32'|'int32' dtype, while the dtype is ${dtype}.`);
   }
   gpuWriteBuffer.unmap();

   const gpuReadBuffer = device.createBuffer({
     mappedAtCreation: false,
     size: sizeInBytes,
     usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE |
         GPUBufferUsage.COPY_SRC
   });

   const copyEncoder = device.createCommandEncoder();
   copyEncoder.copyBufferToBuffer(
       gpuWriteBuffer, 0, gpuReadBuffer, 0, sizeInBytes);
   const copyCommands = copyEncoder.finish();
   device.queue.submit([copyCommands]);
   gpuWriteBuffer.destroy();
   return gpuReadBuffer;
}

const savedBackend = tf.getBackend();
await tf.setBackend('webgpu').catch(
     () => {throw new Error(
// The rest of the code for WebGPU tensor creation would follow here,
// including getting the WebGPU device and calling tf.tensor with
// WebGPUData.
     });

```

--------------------------------

### Get TensorFlow.js Environment

Source: https://js.tensorflow.org/api/latest/index

Returns the current TensorFlow.js environment, which is a global singleton. The environment object provides access to evaluated feature values and the active platform.

```javascript
const env = tf.env ();
```

--------------------------------

### Mirror Pad Tensor in JavaScript ('reflect')

Source: https://js.tensorflow.org/api/latest/index

Pads a tensor using mirror padding. This example uses the 'reflect' mode, where values are reflected across the edges of the tensor.

```javascript
const x = tf.range(0, 9).reshape([1, 1, 3, 3]);
x.mirrorPad([[0, 0], [0, 0], [2, 2], [2, 2]], 'reflect').print();
```

--------------------------------

### Get TensorFlow.js Engine

Source: https://js.tensorflow.org/api/latest/index

Retrieves the global engine instance responsible for managing tensors and backends within TensorFlow.js. This engine is a singleton.

```javascript
const engine = tf.engine ();
```

--------------------------------

### Send Model to an HTTP Server

Source: https://js.tensorflow.org/api/latest/index

This example illustrates how to send a model's topology and weights to a remote HTTP server using the 'http://' URL shortcut. This requires a server-side implementation to handle the incoming model artifacts, suitable for centralized model management.

```javascript
const model = tf.sequential(
     {layers: [tf.layers.dense({units: 1, inputShape: [3]})}]});
const saveResults = await model.save('http://my-server/model/upload');

```

--------------------------------

### Concatenating Datasets in JavaScript

Source: https://js.tensorflow.org/api/latest/index

Concatenates this `Dataset` with another. This example shows how to combine two datasets sequentially.

```javascript
const a = tf.data.array([1, 2, 3]);
const b = tf.data.array([4, 5, 6]);
const c = a.concatenate(b);
await c.forEachAsync(e => console.log(e));
```

--------------------------------

### Construct RNN Layer with Stacked GRUCells - JavaScript

Source: https://js.tensorflow.org/api/latest/index

Shows how to construct an `RNN` layer using a list of `GRUCell` instances, forming a stacked RNN cell. This example creates an input with 10 time steps and a 20-dimensional vector at each step, and the output shape reflects the sequence length and the units of the last `GRUCell` when `returnSequences` is true.

```javascript
const cells = [
   tf.layers.gruCell({units: 4}),
   tf.layers.gruCell({units: 8}),
];
const rnn = tf.layers.rnn({cell: cells, returnSequences: true});

// Create an input with 10 time steps and a length-20 vector at each step.
const input = tf.input({shape: [10, 20]});
const output = rnn.apply(input);

console.log(JSON.stringify(output.shape));
// [null, 10, 8]: 1st dimension is unknown batch size; 2nd dimension is the
// same as the sequence length of `input`, due to `returnSequences`: `true`;
// 3rd dimension is the last `gruCell`'s number of units.
```

--------------------------------

### Batching a Dataset of Numbers in JavaScript

Source: https://js.tensorflow.org/api/latest/index

Groups elements of a dataset into batches. This example demonstrates batching a dataset of numbers. It assumes each incoming element has the same structure. Primitives are grouped into a 1-D Tensor.

```javascript
const a = tf.data.array([1, 2, 3, 4, 5, 6, 7, 8]).batch(4);
await a.forEachAsync(e => e.print());
```

--------------------------------

### Batching a Dataset of Objects in JavaScript

Source: https://js.tensorflow.org/api/latest/index

Groups elements of a dataset into batches. This example demonstrates batching a dataset where elements are objects. For each key, the resulting Dataset provides a batched element collecting all incoming values for that key.

```javascript
const c = tf.data.array([{a: 1, b: 11}, {a: 2, b: 12}, {a: 3, b: 13},
   {a: 4, b: 14}, {a: 5, b: 15}, {a: 6, b: 16}, {a: 7, b: 17},
   {a: 8, b: 18}]).batch(4);
await c.forEachAsync(e => {
   console.log('{');
   for(var key in e) {
     console.log(key+':');
     e[key].print();
   }
   console.log('}');
})
```

--------------------------------

### Create Tensor from Pixels Asynchronously (Browser JavaScript)

Source: https://js.tensorflow.org/api/latest/index

Asynchronously creates a `tf.Tensor` from pixel data, similar to `tf.browser.fromPixels`. This API checks for a `WRAP_TO_IMAGEBITMAP` flag and may wrap the input to `ImageBitmap` for efficiency. The example shows async creation and printing of a tensor from `ImageData`.

```javascript
tf.browser.fromPixelsAsync (pixels, numChannels?) function Source
Creates a tf.Tensor from an image in async way.
```
const image = new ImageData(1, 1);
image.data[0] = 100;
image.data[1] = 150;
image.data[2] = 200;
image.data[3] = 255;

(await tf.browser.fromPixelsAsync(image)).print();

```

EditRun
  * webgpu
  * webgl
  * wasm
  * cpu

This API is the async version of fromPixels. The API will first check |WRAP_TO_IMAGEBITMAP| flag, and try to wrap the input to imageBitmap if the flag is set to true.
Parameters:
  * pixels (PixelData|ImageData|HTMLImageElement|HTMLCanvasElement| HTMLVideoElement|ImageBitmap) The input image to construct the tensor from. The supported image types are all 4-channel. You can also pass in an image object with following attributes: `{data: Uint8Array; width: number; height: number}`
  * numChannels (number) The number of channels of the output tensor. A numChannels value less than 4 allows you to ignore channels. Defaults to 3 (ignores alpha channel of input image). Optional 

Returns: Promise<tf.Tensor>
```

--------------------------------

### Stacked LSTM Cells with RNN Layer - TensorFlow.js

Source: https://js.tensorflow.org/api/latest/index

Illustrates how to construct an RNN layer using multiple LSTMCell instances. This example demonstrates stacking LSTM cells to create a more complex recurrent network and processing sequential data. The output shape is explained in the context of stacked cells and the returnSequences parameter.

```javascript
const cells = [
   tf.layers.lstmCell({units: 4}),
   tf.layers.lstmCell({units: 8}),
];
const rnn = tf.layers.rnn({cell: cells, returnSequences: true});

// Create an input with 10 time steps and a length-20 vector at each step.
const input = tf.input({shape: [10, 20]});
const output = rnn.apply(input);

console.log(JSON.stringify(output.shape));
// [null, 10, 8]: 1st dimension is unknown batch size; 2nd dimension is the
// same as the sequence length of `input`, due to `returnSequences`: `true`;
// 3rd dimension is the last `lstmCell`'s number of units.
```

--------------------------------

### Handling Input Shape and Batch Size for RNN Layers in TensorFlow.js

Source: https://js.tensorflow.org/api/latest/index

This example demonstrates how to specify input dimensions and batch sizes for RNN layers in TensorFlow.js. It covers `inputDim`, `inputLength`, `inputShape`, `batchInputShape`, and `batchSize` parameters, which are crucial for defining the expected shape of input data, especially for the first layer in a model.

```javascript
const layer1 = tf.layers.simpleRNN({
  units: 32,
  inputDim: 10,
  inputLength: 5
});

const layer2 = tf.layers.simpleRNN({
  units: 32,
  inputShape: [5, 10] // [timesteps, inputDim]
});

const layer3 = tf.layers.simpleRNN({
  units: 32,
  batchInputShape: [64, 5, 10] // [batchSize, timesteps, inputDim]
});

const layer4 = tf.layers.simpleRNN({
  units: 32,
  batchSize: 64,
  inputShape: [5, 10]
});
```

--------------------------------

### Get Current Backend with tf.backend

Source: https://js.tensorflow.org/api/latest/index

Retrieves the currently active backend for TensorFlow.js. If no backend is initialized, it attempts to initialize the best available one. This function may throw an error if the chosen backend requires asynchronous initialization.

```javascript
// Assuming tf.ready() has been awaited if necessary
const currentBackend = tf.backend();
console.log(currentBackend);
```

--------------------------------

### Add Layers to a Sequential Model in JavaScript

Source: https://js.tensorflow.org/api/latest/index

Illustrates how to build a multi-layer sequential model by progressively adding dense layers with specified units, input shapes, and activation functions. The example shows adding three dense layers and then performing prediction on random input data.

```javascript
  const model = tf.sequential();
  model.add(tf.layers.dense({units: 8, inputShape: [1]}));
  model.add(tf.layers.dense({units: 4, activation: 'relu6'}));
  model.add(tf.layers.dense({units: 1, activation: 'relu6'}));
  // Note that the untrained model is random at this point.
  model.predict(tf.randomNormal([10, 1])).print();

```

--------------------------------

### tf.layers.gru Example: Applying a GRU layer to sequential input

Source: https://js.tensorflow.org/api/latest/index

This JavaScript code snippet demonstrates how to create and apply a GRU layer using TensorFlow.js. It defines a GRU layer with 8 units and configures it to return sequences. An input tensor with a shape of [10, 20] is then created, and the GRU layer is applied to it. The output shape is logged to the console, showing the result of processing sequential data.

```javascript
const rnn = tf.layers.gru({units: 8, returnSequences: true});

// Create an input with 10 time steps.
const input = tf.input({shape: [10, 20]});
const output = rnn.apply(input);

console.log(JSON.stringify(output.shape));
// [null, 10, 8]: 1st dimension is unknown batch size; 2nd dimension is the
// same as the sequence length of `input`, due to `returnSequences`: `true`;
// 3rd dimension is the `GRUCell`'s number of units.
```

--------------------------------

### Get Current Backend Name with tf.getBackend

Source: https://js.tensorflow.org/api/latest/index

Returns the name of the currently active backend (e.g., 'cpu', 'webgl'). The backend is responsible for tensor creation and operation execution.

```javascript
const backendName = tf.getBackend();
console.log(backendName);
```

--------------------------------

### Batching a Dataset of Arrays in JavaScript

Source: https://js.tensorflow.org/api/latest/index

Groups elements of a dataset into batches. This example shows batching a dataset where elements are arrays. Incoming arrays are converted to Tensors and then batched.

```javascript
const b = tf.data.array([[1], [2], [3], [4], [5], [6], [7], [8]]).batch(4);
await b.forEachAsync(e => e.print());
```

--------------------------------

### Initialize Tensors with Zeros in TensorFlow.js

Source: https://js.tensorflow.org/api/latest/index

The tf.initializers.zeros() function creates an initializer that generates tensors filled with zeros. This is a common way to start model weights or tensor values.

```javascript
tf.initializers.zeros ()
```

--------------------------------

### Get Value from tf.TensorBuffer (JavaScript)

Source: https://js.tensorflow.org/api/latest/index

The `get()` method of a tf.TensorBuffer retrieves the value stored at a given location within the buffer.

```javascript
const value = buffer.get(loc1, loc2, ...);
```

--------------------------------

### Zip Dictionary of Datasets using tf.data.zip

Source: https://js.tensorflow.org/api/latest/index

Creates a Dataset by zipping together a dictionary of Datasets. The resulting dataset produces elements that are dictionaries where keys are from the input dictionary and values are elements from the corresponding datasets. This example demonstrates zipping two datasets into a dictionary.

```javascript
const a = tf.data.array([{a: 1}, {a: 2}, {a: 3}]);
const b = tf.data.array([{b: 4}, {b: 5}, {b: 6}]);
const c = tf.data.zip({c: a, d: b});
await c.forEachAsync(e => console.log(JSON.stringify(e)));

```

--------------------------------

### Iterating Over a Dataset Asynchronously in JavaScript

Source: https://js.tensorflow.org/api/latest/index

Applies a function to every element of the dataset asynchronously. After the function is applied, any Tensors contained within the element are disposed. This example prints each element of the dataset.

```javascript
const a = tf.data.array([1, 2, 3]);
await a.forEachAsync(e => console.log(e));
```

--------------------------------

### Get Tensor as Human-Readable String (JavaScript)

Source: https://js.tensorflow.org/api/latest/index

The `toString()` method provides a human-readable string representation of the tensor, which is helpful for debugging and logging. An optional verbose parameter can be provided for more detailed output.

```javascript
const tensorString = tensor.toString();
const verboseString = tensor.toString(true);
```

--------------------------------

### Construct RNN Layer with Stacked SimpleRNNCells in TensorFlow.js

Source: https://js.tensorflow.org/api/latest/index

Illustrates the construction of an RNN layer using stacked SimpleRNNCells in TensorFlow.js. This example shows how to combine multiple cells and configure the RNN layer to return sequences. The output shape reflects the batch size, sequence length, and the units of the last cell.

```javascript
const cells = [
   tf.layers.simpleRNNCell({units: 4}),
   tf.layers.simpleRNNCell({units: 8}),
];
const rnn = tf.layers.rnn({cell: cells, returnSequences: true});

// Create an input with 10 time steps and a length-20 vector at each step.
const input = tf.input({shape: [10, 20]});
const output = rnn.apply(input);

console.log(JSON.stringify(output.shape));
// [null, 10, 8]: 1st dimension is unknown batch size; 2nd dimension is the
// same as the sequence length of `input`, due to `returnSequences`: `true`;
```

--------------------------------

### Convert Depth to Space in JavaScript (NHWC)

Source: https://js.tensorflow.org/api/latest/index

Rearranges data from the depth dimension into spatial blocks for a rank 4 tensor. This example uses the 'NHWC' data format.

```javascript
const x = tf.tensor4d([1, 2, 3, 4], [1, 1, 1, 4]);
const blockSize = 2;
const dataFormat = "NHWC";

tf.depthToSpace(x, blockSize, dataFormat).print();
```

--------------------------------

### Filtering a Dataset in JavaScript

Source: https://js.tensorflow.org/api/latest/index

Filters this dataset according to a predicate function. This example filters a dataset to keep only even numbers.

```javascript
const a = tf.data.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
   .filter(x => x%2 === 0);
await a.forEachAsync(e => console.log(e));
```

--------------------------------

### Slice Tensor1D using TensorFlow.js

Source: https://js.tensorflow.org/api/latest/index

Extracts a slice from a 1-dimensional tensor. This TensorFlow.js function takes a tensor, a starting point (begin), and a size for the slice. It's useful for selecting subsets of tensor data. Supports various execution backends.

```javascript
const x = tf.tensor1d([1, 2, 3, 4]);

x.slice([1], [2]).print();

```

--------------------------------

### Create Scalar Tensor with tf.scalar

Source: https://js.tensorflow.org/api/latest/index

Illustrates the creation of a rank-0 tensor (scalar) using the `tf.scalar` function. This is a more readable alternative to `tf.tensor` for single-value tensors. The example shows how to print the scalar's value.

```javascript
tf.scalar(3.14).print();
```

--------------------------------

### Get Values and Gradients of a Function with tf.valueAndGrads

Source: https://js.tensorflow.org/api/latest/index

Extends `tf.grads()` by also returning the value of the function `f`. Useful for displaying metrics alongside gradients. It returns an object containing the gradients for each input (`grads`) and the function's output value (`value`). Supported on webgpu, webgl, wasm, and cpu.

```javascript
const f = (a, b) => a.mul(b);
const g = tf.valueAndGrads(f);

const a = tf.tensor1d([2, 3]);
const b = tf.tensor1d([-2, -3]);
const {value, grads} = g([a, b]);

const [da, db] = grads;

console.log('value');
value.print();

console.log('da');
da.print();
console.log('db');
db.print();

```

--------------------------------

### Save Model as Browser Downloadable Files

Source: https://js.tensorflow.org/api/latest/index

This snippet demonstrates saving a model's topology and weights as two separate files (`.json` for topology and `.weights.bin` for weights) that are automatically downloaded by the browser using the 'downloads://' URL shortcut. This is convenient for manual distribution or backup.

```javascript
const model = tf.sequential(
     {layers: [tf.layers.dense({units: 1, inputShape: [3]})}]});
const saveResults = await model.save('downloads://my-model-1');

```

--------------------------------

### Configure 1D Convolutional Layer with Strides and Padding in TensorFlow.js

Source: https://js.tensorflow.org/api/latest/index

This example shows how to configure a tf.layers.conv1d layer with specific strides and padding. Strides control the step size of the convolution, and padding determines how the input is extended. This is crucial for controlling the output dimensions.

```javascript
const convLayerWithStrides = tf.layers.conv1d({
  filters: 64,
  kernelSize: 5,
  strides: 2, // Move the kernel 2 steps at a time
  padding: 'same', // Pad the input so the output has the same spatial dimensions
  activation: 'tanh',
  inputShape: [null, 64] // Variable-length sequences of 64-dimensional vectors
});
console.log(convLayerWithStrides);
```

--------------------------------

### Get Memory Information with tf.memory

Source: https://js.tensorflow.org/api/latest/index

Returns memory info at the current time in the program. The result is an object with properties like numBytes, numTensors, and numDataBuffers. For WebGL, it also includes numBytesInGPU.

```javascript
// Example usage for tf.memory() would involve calling it directly:
// const memInfo = tf.memory();
// console.log(memInfo);
// The actual code example is conceptual and demonstrates the return object structure.
```

--------------------------------

### tf.ready

Source: https://js.tensorflow.org/api/latest/index

Returns a promise that resolves when the currently selected backend has initialized.

```APIDOC
## tf.ready ()

### Description
Returns a promise that resolves when the currently selected backend (or the highest priority one) has initialized. Await this promise when you are using a backend that has async initialization.

### Method
N/A (Backend function)

### Endpoint
N/A

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
None

### Request Example
```javascript
async function runWithBackend() {
  await tf.ready();
  console.log('Backend is ready.');
  // Perform TensorFlow.js operations
}
runWithBackend();
```

### Response
#### Success Response (N/A)
- **Promise<void>** - A promise that resolves when the backend is initialized.

#### Response Example
```json
// No direct JSON response, the promise resolves.
```
```

--------------------------------

### Create and Train Sequential Model in JavaScript

Source: https://js.tensorflow.org/api/latest/index

Demonstrates how to create a sequential model, add a dense layer for linear regression, compile it with a loss function and optimizer, train it with synthetic data, and perform inference. This involves using `tf.sequential()`, `model.add()`, `model.compile()`, `tf.tensor2d()`, `model.fit()`, and `model.predict()`.

```javascript
  // Define a model for linear regression.
  const model = tf.sequential();
  model.add(tf.layers.dense({units: 1, inputShape: [1]}));

  // Prepare the model for training: Specify the loss and the optimizer.
  model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

  // Generate some synthetic data for training.
  const xs = tf.tensor2d([1, 2, 3, 4], [4, 1]);
  const ys = tf.tensor2d([1, 3, 5, 7], [4, 1]);

  // Train the model using the data then do inference on a data point the
  // model hasn't seen:
  await model.fit(xs, ys);
  model.predict(tf.tensor2d([5], [1, 1])).print();

```

--------------------------------

### Get Value and Gradient of a Function with tf.valueAndGrad

Source: https://js.tensorflow.org/api/latest/index

Similar to `tf.grad()`, but also returns the value of the function `f`. This is useful when `f` returns a metric that needs to be displayed. It returns an object containing both the gradient (`grad`) and the value (`value`). Available for webgpu, webgl, wasm, and cpu.

```javascript
const f = x => x.square();
const g = tf.valueAndGrad(f);

const x = tf.tensor1d([2, 3]);
const {value, grad} = g(x);

console.log('value');
value.print();
console.log('grad');
grad.print();

```

--------------------------------

### Zip Array of Datasets using tf.data.zip

Source: https://js.tensorflow.org/api/latest/index

Creates a Dataset by zipping together an array of Datasets. The resulting dataset produces elements that are arrays of elements from the input datasets. The number of elements is determined by the smallest dataset. This example demonstrates zipping two datasets of objects and then merging them.

```javascript
console.log('Zip two datasets of objects:');
const ds1 = tf.data.array([{a: 1}, {a: 2}, {a: 3}]);
const ds2 = tf.data.array([{b: 4}, {b: 5}, {b: 6}]);
const ds3 = tf.data.zip([ds1, ds2]);
await ds3.forEachAsync(e => console.log(JSON.stringify(e)));

// If the goal is to merge the dicts in order to produce elements like
// {a: ..., b: ...}, this requires a second step such as:
console.log('Merge the objects:');
const ds4 = ds3.map(x => {return {a: x[0].a, b: x[1].b}});
await ds4.forEachAsync(e => console.log(e));

```

--------------------------------

### Get Current High-Resolution Time with tf.util.now

Source: https://js.tensorflow.org/api/latest/index

Returns the current high-resolution time in milliseconds relative to an arbitrary past time. This function is cross-platform, working in both Node.js and browser environments.

```javascript
console.log(tf.util.now());
```

--------------------------------

### Instantiate and Use ConvLSTM2DCell - JavaScript

Source: https://js.tensorflow.org/api/latest/index

This snippet demonstrates how to instantiate a ConvLSTM2DCell, build it with an input shape, and then call it with sample input data and initial states. It highlights the cell's input and output structure for a single time step.

```javascript
const filters = 3;
const kernelSize = 3;

const sequenceLength = 1;
const size = 5;
const channels = 3;

const inputShape = [sequenceLength, size, size, channels];
const input = tf.ones(inputShape);

const cell = tf.layers.convLstm2dCell({
  filters,
  kernelSize
});

cell.build(input.shape);

const outputSize = size - kernelSize + 1;
const outShape = [sequenceLength, outputSize, outputSize, filters];

const initialH = tf.zeros(outShape);
const initialC = tf.zeros(outShape);

const [o, h, c] = cell.call([input, initialH, initialC], {});

```

--------------------------------

### Create Tensor from WebGL Texture Data in JavaScript

Source: https://js.tensorflow.org/api/latest/index

Creates a tensor directly from WebGL texture data, allowing for GPU-to-GPU data transfer without CPU synchronization. This example demonstrates setting up a custom WebGL backend, creating a WebGL texture, uploading data to it, and then creating a TensorFlow.js tensor from that texture. The tensor's values are printed, and the original texture data can be retrieved using `dataToGPU`.

```javascript
// Pass a `WebGLData` object and specify a shape yourself.

// This makes it possible for TF.js applications to avoid GPU / CPU sync.
// For example, if your application includes a preprocessing step on the GPU,
// you could upload the GPU output directly to TF.js, rather than first
// downloading the values.

// Example for WebGL2:
if (tf.findBackend('custom-webgl') == null) {
   const customCanvas = document.createElement('canvas');
   const customBackend = new tf.MathBackendWebGL(customCanvas);
   tf.registerBackend('custom-webgl', () => customBackend);
}
const savedBackend = tf.getBackend();
await tf.setBackend('custom-webgl');
const gl = tf.backend().gpgpu.gl;
const texture = gl.createTexture();
const tex2d = gl.TEXTURE_2D;
const width = 2;
const height = 2;

gl.bindTexture(tex2d, texture);
gl.texParameteri(tex2d, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
gl.texParameteri(tex2d, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
gl.texParameteri(tex2d, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
gl.texParameteri(tex2d, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
gl.texImage2D(
   tex2d, 0, gl.RGBA32F, // internalFormat
   width, height, 0,
   gl.RGBA, // textureFormat
   gl.FLOAT, // textureType
   new Float32Array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
);

// Currently, the `texture` has 4 pixels:
// Pixel0 is {R:0, G:1, B:2, A:3}
// Pixel1 is {R:4, G:5, B:6, A:7}
// Pixel2 is {R:8, G:9, B:10, A:11}
// Pixel3 is {R:12, G:13, B:14, A:15}

const logicalShape = [height * width * 2];
const a = tf.tensor({texture, height, width, channels: 'BR'}, logicalShape);
a.print();
// Tensor value will be [2, 0, 6, 4, 10, 8, 14, 12], since [2, 0] is the
// values of 'B' and 'R' channels of Pixel0, [6, 4] is the values of 'B' and
'R'
// channels of Pixel1...

// For postprocessing on the GPU, it's possible to retrieve the texture
// backing any tensor by calling the tensor's `dataToGPU` method like
// so:

const tex = a.dataToGPU();
await tf.setBackend(savedBackend);

```

--------------------------------

### Early Stopping Callback for Model Training in TensorFlow.js

Source: https://js.tensorflow.org/api/latest/index

Implements an early stopping callback to prevent overfitting during model training. It monitors a specified metric (e.g., 'val_acc') and stops training if the metric stops improving for a defined number of epochs ('patience'). This example demonstrates its use with `tf.LayersModel.fit()`, supporting webgpu, webgl, wasm, and cpu.

```javascript
const model = tf.sequential();
model.add(tf.layers.dense({
   units: 3,
   activation: 'softmax',
   kernelInitializer: 'ones',
   inputShape: [2]
}));
const xs = tf.tensor2d([1, 2, 3, 4], [2, 2]);
const ys = tf.tensor2d([[1, 0, 0], [0, 1, 0]], [2, 3]);
const xsVal = tf.tensor2d([4, 3, 2, 1], [2, 2]);
const ysVal = tf.tensor2d([[0, 0, 1], [0, 1, 0]], [2, 3]);
model.compile(
     {loss: 'categoricalCrossentropy', optimizer: 'sgd', metrics: ['acc']});

// Without the EarlyStopping callback, the val_acc value would be:
//   0.5, 0.5, 0.5, 0.5, ...
// With val_acc being monitored, training should stop after the 2nd epoch.
const history = await model.fit(xs, ys, {
   epochs: 10,
   validationData: [xsVal, ysVal],
   callbacks: tf.callbacks.earlyStopping({monitor: 'val_acc'})
});

// Expect to see a length-2 array.
console.log(history.history.val_acc);
```

--------------------------------

### Create and Use InputLayer in TensorFlow.js

Source: https://js.tensorflow.org/api/latest/index

Demonstrates how to create an `InputLayer` in TensorFlow.js, which serves as an entry point for models. This is particularly useful when constructing sequential models from existing layers or subsets of other models, ensuring compatibility by explicitly defining the input shape.

```javascript
const model1 = tf.sequential();
model1.add(tf.layers.dense({inputShape: [4], units: 3, activation: 'relu'}));
model1.add(tf.layers.dense({units: 1, activation: 'sigmoid'}));
model1.summary();
model1.predict(tf.zeros([1, 4])).print();

const model2 = tf.sequential();
// Use an inputShape that matches the input shape of `model1`'s second
// layer.
model2.add(tf.layers.inputLayer({inputShape: [3]}));
model2.add(model1.layers[1]);
model2.summary();
model2.predict(tf.zeros([1, 3])).print();
```

--------------------------------

### Create 1D Tensor with tf.tensor1d

Source: https://js.tensorflow.org/api/latest/index

Demonstrates creating a rank-1 tensor (vector) using the `tf.tensor1d` function. This function provides a convenient way to create 1D tensors from arrays or TypedArrays, offering better readability than `tf.tensor`. The example prints the tensor's values.

```javascript
tf.tensor1d([1, 2, 3]).print();
```

--------------------------------

### Slice Tensor2D using TensorFlow.js

Source: https://js.tensorflow.org/api/latest/index

Extracts a slice from a 2-dimensional tensor. This TensorFlow.js function allows for precise selection of sub-tensors by specifying the starting coordinates and the size of the slice along each dimension. It is compatible with multiple backends.

```javascript
const x = tf.tensor2d([1, 2, 3, 4], [2, 2]);

x.slice([1, 0], [1, 2]).print();

```

--------------------------------

### Build Model with Symbolic Tensors

Source: https://js.tensorflow.org/api/latest/index

Illustrates building a model using `tf.layers.Layer` with `tf.SymbolicTensor` inputs. This is typically used for defining the structure of non-Sequential models. It shows how to obtain a `SymbolicTensor` using `tf.input`, apply layers sequentially to it, and infer output shapes, finally constructing a `tf.model`.

```javascript
const flattenLayer = tf.layers.flatten();
const denseLayer = tf.layers.dense({units: 1});

// Use tf.layers.input() to obtain a SymbolicTensor as input to apply().
const input = tf.input({shape: [2, 2]});
const output1 = flattenLayer.apply(input);

// output1.shape is [null, 4]. The first dimension is the undetermined
// batch size. The second dimension comes from flattening the [2, 2]
// shape.
console.log(JSON.stringify(output1.shape));

// The output SymbolicTensor of the flatten layer can be used to call
// the apply() of the dense layer:
const output2 = denseLayer.apply(output1);

// output2.shape is [null, 1]. The first dimension is the undetermined
// batch size. The second dimension matches the number of units of the
// dense layer.
console.log(JSON.stringify(output2.shape));

// The input and output can be used to construct a model that consists
// of the flatten and dense layers.
const model = tf.model({inputs: input, outputs: output2});

```

--------------------------------

### Register Custom TensorFlow Op in JavaScript

Source: https://js.tensorflow.org/api/latest/index

Shows how to register a custom operation (Op) for the TensorFlow.js graph model executor. This example defines a custom `MatMul` operation that uses `tf.matMul` and then registers it using `tf.registerOp()`. This allows overriding existing ops or adding new ones.

```javascript
const customMatmul = (node) =>
    tf.matMul(
        node.inputs[0], node.inputs[1],
        node.attrs['transpose_a'], node.attrs['transpose_b']);

tf.registerOp('MatMul', customMatmul);

```

--------------------------------

### Export Model to Browser Downloads with TensorFlow.js

Source: https://js.tensorflow.org/api/latest/index

This snippet demonstrates how to export a TensorFlow.js model and trigger a download in the browser. It uses the `tf.io.browserDownloads` function, which creates an IOHandler that initiates file downloads. The downloaded files include the model's topology ('model.json') and weights ('model.weights.bin'), with an optional prefix for file naming. This is useful for saving trained models directly from the browser.

```javascript
const model = tf.sequential();
model.add(tf.layers.dense({units: 1, inputShape: [10], activation: 'sigmoid'}));
const saveResult = await model.save('downloads://mymodel');
// This will trigger downloading of two files:
//   'mymodel.json' and 'mymodel.weights.bin'.
console.log(saveResult);
```

--------------------------------

### Calculate Binary Accuracy (JavaScript)

Source: https://js.tensorflow.org/api/latest/index

Computes the binary accuracy metric between true labels (`yTrue`) and predicted labels (`yPred`). Both tensors are expected to contain values between 0 and 1. The example demonstrates calculating accuracy for two sets of predictions.

```javascript
tf.metrics.binaryAccuracy (yTrue, yPred) function Source
Binary accuracy metric function.
`yTrue` and `yPred` can have 0-1 values. Example:
```
const x = tf.tensor2d([[1, 1, 1, 1], [0, 0, 0, 0]], [2, 4]);
const y = tf.tensor2d([[1, 0, 1, 0], [0, 0, 0, 1]], [2, 4]);

```

--------------------------------

### Defining RNN Cell and Input Shape in TensorFlow.js

Source: https://js.tensorflow.org/api/latest/index

This example shows how to define an RNN layer with a specific cell type and input dimensions. It highlights the `cell` parameter, which accepts a `tf.RNNCell` instance, and input shape parameters like `inputDim` or `inputShape` required for the first layer in a model.

```javascript
const rnnCell = tf.layers.simpleRNN({
  units: 16,
  returnSequences: true,
  inputDim: 8 // Dimensionality of the input
});

const model = tf.sequential({
  layers: [
    rnnCell
  ]
});

// Alternatively, using inputShape:
const rnnCellWithInputShape = tf.layers.simpleRNN({
  units: 16,
  returnSequences: true,
  inputShape: [10, 8] // Shape: [timesteps, inputDim]
});

const model2 = tf.sequential({
  layers: [
    rnnCellWithInputShape
  ]
});
```

--------------------------------

### Load Model from Browser Files with TensorFlow.js

Source: https://js.tensorflow.org/api/latest/index

This snippet illustrates how to load model artifacts from user-selected files in the browser using `tf.io.browserFiles`. This IOHandler is suitable for loading models when the model's topology (JSON) and weights (binary) are provided as files. When used with `tf.loadLayersModel`, it allows constructing a Keras-style `tf.LayersModel` from these loaded artifacts. Note that this code requires corresponding HTML file input elements.

```javascript
// Note: This code snippet won't run properly without the actual file input
//   elements in the HTML DOM.

// Suppose there are two HTML file input (`<input type="file" ...>`)
// elements.
const uploadJSONInput = document.getElementById('upload-json');
const uploadWeightsInput = document.getElementById('upload-weights');
const model = await tf.loadLayersModel(tf.io.browserFiles([
uploadJSONInput,
uploadWeightsInput
]));
```

--------------------------------

### Backend Management API

Source: https://js.tensorflow.org/api/latest/index

APIs for managing TensorFlow.js backends, including removing and setting the active backend.

```APIDOC
## tf.removeBackend

### Description
Removes a backend and its registered factory.

### Method
`tf.removeBackend(name)`

### Parameters
#### Path Parameters
- **name** (string) - The name of the backend to remove.

### Returns
void

## tf.setBackend

### Description
Sets the backend responsible for creating tensors and executing operations. This operation disposes of the current backend and its associated tensors. A new backend is initialized, even if it's of the same type.

### Method
`tf.setBackend(backendName)`

### Parameters
#### Path Parameters
- **backendName** (string) - The name of the backend to set. Supported values include `'webgl'`, `'cpu'`, `'wasm'`, and `'tensorflow'` (for Node.js).

### Returns
Promise<boolean> - A promise that resolves to `true` if the backend initialization was successful, `false` otherwise.
```

--------------------------------

### prefetch (bufferSize)

Source: https://js.tensorflow.org/api/latest/index

Creates a `Dataset` that prefetches elements from this dataset.

```APIDOC
## prefetch (bufferSize)

### Description
Creates a `Dataset` that prefetches elements from this dataset.

### Method
prefetch

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **bufferSize** (number) - An integer specifying the number of elements to be prefetched.

### Request Example
```javascript
// Example not provided in source, but conceptually:
// const dataset = tf.data.array([1, 2, 3]).prefetch(2);
```

### Response
#### Success Response (200)
- **tf.data.Dataset** - The dataset with prefetched elements.

#### Response Example
```json
{
  "example": "Dataset object"
}
```
```

--------------------------------

### Apply Activation Layer and Create Model in TensorFlow.js

Source: https://js.tensorflow.org/api/latest/index

This snippet demonstrates applying an activation layer to a dense layer's output and then creating a TensorFlow.js model with both the dense output and the activated output. It also shows how to predict using the model and print the results.

```javascript
const activationOutput = activationLayer.apply(denseOutput);

// Create the model based on the inputs.
const model = tf.model({
     inputs: input,
     outputs: [denseOutput, activationOutput]
});

// Collect both outputs and print separately.
const [denseOut, activationOut] = model.predict(tf.randomNormal([6, 5]));
denseOut.print();
activationOut.print();
```

--------------------------------

### Manage TensorFlow Backends (JavaScript)

Source: https://js.tensorflow.org/api/latest/index

Functions to manage the backend responsible for tensor operations. `setBackend` initializes or switches the backend (e.g., 'webgl', 'cpu', 'wasm'), potentially disposing of the current one. `removeBackend` unregisters a backend factory.

```javascript
tf.removeBackend (name) function Source
Removes a backend and the registered factory.
Parameters:
  * name (string) 
```

```javascript
tf.setBackend (backendName) function Source
Sets the backend (cpu, webgl, wasm, etc) responsible for creating tensors and executing operations on those tensors. Returns a promise that resolves to a boolean if the backend initialization was successful.
Note this disposes the current backend, if any, as well as any tensors associated with it. A new backend is initialized, even if it is of the same type as the previous one.
Parameters:
  * backendName (string) The name of the backend. Currently supports 'webgl'|'cpu' in the browser, 'tensorflow' under node.js (requires tfjs-node), and 'wasm' (requires tfjs-backend-wasm).
Returns: Promise<boolean>
```

--------------------------------

### Create Tensor with Numbers in Range

Source: https://js.tensorflow.org/api/latest/index

Generates a 1D tensor containing numbers within a specified range. The range is half-open, including the start value but excluding the stop value. Supports decremented ranges and negative steps. This function is compatible with webgpu, webgl, wasm, and cpu.

```javascript
tf.range(0, 9, 2).print();
```

--------------------------------

### tf.signal.hannWindow

Source: https://js.tensorflow.org/api/latest/index

Generate a Hann window.

```APIDOC
## POST /tf.signal.hannWindow

### Description
Generate a Hann window.
See: https://en.wikipedia.org/wiki/Window_function#Hann_and_Hamming_windows

### Method
POST

### Endpoint
/tf.signal.hannWindow

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **windowLength** (number) - Required - The length of the window.

### Request Example
```json
{
  "windowLength": 10
}
```

### Response
#### Success Response (200)
- **tensor** (tf.Tensor1D) - The generated Hann window tensor.

#### Response Example
```json
{
  "tensor": "[Tensor object representation]"
}
```
```

--------------------------------

### take (count)

Source: https://js.tensorflow.org/api/latest/index

Creates a `Dataset` with at most `count` initial elements from this dataset.

```APIDOC
## take (count)

### Description
Creates a `Dataset` with at most `count` initial elements from this dataset.

### Method
take

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **count** (number) - The number of elements of this dataset that should be taken to form the new dataset. If `count` is `undefined` or negative, or if `count` is greater than the size of this dataset, the new dataset will contain all elements of this dataset.

### Request Example
```javascript
const a = tf.data.array([1, 2, 3, 4, 5, 6]).take(3);
a.forEachAsync(e => console.log(e));
```

### Response
#### Success Response (200)
- **tf.data.Dataset** - The dataset with a limited number of elements.

#### Response Example
```json
{
  "example": "Dataset object"
}
```
```

--------------------------------

### Concatenate Tensors (1D) - JavaScript

Source: https://js.tensorflow.org/api/latest/index

Concatenates a list of 1D tensors along a given axis. The tensors must have matching types and sizes in all dimensions except the concatenation axis. This example demonstrates concatenation without an explicit axis, defaulting to the first dimension.

```javascript
const a = tf.tensor1d([1, 2]);
const b = tf.tensor1d([3, 4]);
a.concat(b).print();  // or a.concat(b)
```

```javascript
const a = tf.tensor1d([1, 2]);
const b = tf.tensor1d([3, 4]);
const c = tf.tensor1d([5, 6]);
tf.concat([a, b, c]).print();
```

--------------------------------

### tf.registerBackend

Source: https://js.tensorflow.org/api/latest/index

Registers a global backend.

```APIDOC
## tf.registerBackend (name, factory, priority?)

### Description
Registers a global backend. The registration should happen when importing a module file (e.g. when importing `backend_webgl.ts`), and is used for modular builds (e.g. custom tfjs bundle with only webgl support).

### Method
N/A (Backend function)

### Endpoint
N/A

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
None

### Request Example
```javascript
// Example of registering a custom backend (hypothetical)
const customBackendFactory = () => {
  return {
    // ... backend implementation details ...
    name: 'custom',
    binding: () => ({ name: 'custom' })
  };
};
tf.registerBackend('custom', customBackendFactory, 2);
console.log('Custom backend registered.');
```

### Response
#### Success Response (N/A)
- **boolean** - True if the backend was successfully registered, false otherwise.

#### Response Example
```json
true
```
```

--------------------------------

### tf.inTopKAsync: Check if Targets are in Top K Predictions

Source: https://js.tensorflow.org/api/latest/index

Returns a boolean tensor indicating whether the targets are within the top K predictions for each example. It takes predictions, targets, and an optional K value as input. Supports multiple backends like webgpu, webgl, wasm, and cpu.

```javascript
const predictions = tf.tensor2d([[20, 10, 40, 30], [30, 50, -20, 10]]);
const targets = tf.tensor1d([2, 0]);
const precision = await tf.inTopKAsync(predictions, targets);
precision.print();
```

--------------------------------

### Conv2D Transpose Layer with Input Shape Configuration

Source: https://js.tensorflow.org/api/latest/index

This example shows how to define the input shape directly when creating a tf.layers.conv2dTranspose layer. This is particularly useful when the layer is the first layer in a model. The `inputShape` parameter accepts an array of integers representing the dimensions of the input, excluding the batch axis.

```javascript
const tf = require('@tensorflow/tfjs');

// Example: Creating a transposed convolutional layer with inputShape
const conv2dTransposeWithInputShape = tf.layers.conv2dTranspose({
  filters: 64,
  kernelSize: 5,
  strides: 2,
  padding: 'valid',
  activation: 'sigmoid',
  inputShape: [128, 128, 3] // For 128x128 RGB images
});

console.log(conv2dTransposeWithInputShape.getConfig());
```

--------------------------------

### Register a Custom Backend with tf.registerBackend

Source: https://js.tensorflow.org/api/latest/index

Registers a new backend globally with TensorFlow.js. This is typically done in module files for custom builds. It takes a name, a factory function that returns a backend instance (or a Promise of one), and an optional priority.

```javascript
// Example usage (simplified):
const customBackendFactory = () => {
  // Return a custom backend instance
  return { /* backend implementation */ };
};
tf.registerBackend('my-custom-backend', customBackendFactory, 5);
console.log('Custom backend registered.');
```

--------------------------------

### tf.layers.simpleRNN API

Source: https://js.tensorflow.org/api/latest/index

Documentation for the tf.layers.simpleRNN function, including its parameters and usage.

```APIDOC
## POST /tf.layers.simpleRNN

### Description
Implements a fully-connected Recurrent Neural Network layer where the output is fed back into the input. This layer consists of one `SimpleRNNCell` and operates on a sequence of inputs.

### Method
POST

### Endpoint
/tf.layers.simpleRNN

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **args** (Object) - The configuration object for the SimpleRNN layer.
  - **units** (number) - Positive integer, dimensionality of the output space. Required.
  - **activation** (string) - Activation function to use. Defaults to 'tanh'.
  - **useBias** (boolean) - Whether the layer uses a bias vector. Defaults to true.
  - **kernelInitializer** (string|tf.initializers.Initializer) - Initializer for the `kernel` weights matrix. Defaults to 'glorotUniform'.
  - **recurrentInitializer** (string|tf.initializers.Initializer) - Initializer for the `recurrentKernel` weights matrix. Defaults to 'orthogonal'.
  - **biasInitializer** (string|tf.initializers.Initializer) - Initializer for the bias vector. Defaults to 'zeros'.
  - **kernelRegularizer** (string|Regularizer) - Regularizer function applied to the kernel weights matrix.
  - **recurrentRegularizer** (string|Regularizer) - Regularizer function applied to the recurrentKernel weights matrix.
  - **biasRegularizer** (string|Regularizer) - Regularizer function applied to the bias vector.
  - **kernelConstraint** (string|tf.constraints.Constraint) - Constraint function applied to the kernel weights matrix.
  - **recurrentConstraint** (string|tf.constraints.Constraint) - Constraint function applied to the recurrentKernel weights matrix.
  - **biasConstraint** (string|tf.constraints.Constraint) - Constraint function applied to the bias vector.
  - **dropout** (number) - Fraction of the units to drop for the linear transformation of the inputs (between 0 and 1).
  - **recurrentDropout** (number) - Fraction of the units to drop for the linear transformation of the recurrent state (between 0 and 1).
  - **dropoutFunc** (Function) - Function for dropout, used for test DI purpose.
  - **cell** (tf.RNNCell|tf.RNNCell[]) - A RNN cell instance or a list of RNN cell instances.
  - **returnSequences** (boolean) - Whether to return the last output or the full sequence. Defaults to false.
  - **returnState** (boolean) - Whether to return the last state in addition to the output. Defaults to false.

### Request Example
```json
{
  "units": 8,
  "returnSequences": true
}
```

### Response
#### Success Response (200)
- **layer** (object) - The configured SimpleRNN layer instance.

#### Response Example
```json
{
  "layer": {
    "units": 8,
    "returnSequences": true,
    "activation": "tanh",
    "useBias": true,
    "kernelInitializer": {"config":{"gain":1.0,"mode":"fanDefault"},"className":"glorotUniform"},
    "recurrentInitializer": {"config":{},"className":"orthogonal"},
    "biasInitializer": {"config":{},"className":"zeros"},
    "kernelRegularizer": null,
    "recurrentRegularizer": null,
    "biasRegularizer": null,
    "kernelConstraint": null,
    "recurrentConstraint": null,
    "biasConstraint": null,
    "dropout": 0,
    "recurrentDropout": 0,
    "name": "simple_rnn_1",
    "trainable": true,
    "dtype": "float32",
    "stateful": false,
    "resetAfterInherits": false,
    "returnState": false
  }
}
```
```

--------------------------------

### Apply Conv2D Layer to Each Time Step with TimeDistributed (JavaScript)

Source: https://js.tensorflow.org/api/latest/index

Illustrates using tf.layers.timeDistributed to apply a Conv2D layer to each temporal slice of a 3D input. This example showcases applying convolutional operations across time, which is common in video or sequence analysis where spatial patterns within time steps are relevant. The input and output shapes are logged.

```javascript
const model = tf.sequential();
model.add(tf.layers.timeDistributed({
   layer: tf.layers.conv2d({filters: 64, kernelSize: [3, 3]}),
   inputShape: [10, 299, 299, 3],
}));
console.log(JSON.stringify(model.outputs[0].shape));
```

--------------------------------

### Create One-Hot Encoded Tensor (JavaScript)

Source: https://js.tensorflow.org/api/latest/index

Generates a one-hot encoded tensor. Specified indices are set to `onValue` (defaulting to 1), and others to `offValue` (defaulting to 0). The output tensor has an additional dimension for the depth. Indices must be of type 'int32' and start from 0. Supported backends include webgpu, webgl, wasm, and cpu.

```javascript
tf.oneHot(tf.tensor1d([0, 1], 'int32'), 3).print();
```

--------------------------------

### Create and Add Tensors with GPU Buffer

Source: https://js.tensorflow.org/api/latest/index

Demonstrates creating tensors from a GPU buffer and a flat array, performing element-wise addition, and managing tensor disposal. It highlights the use of `tf.tensor` with a `GPUBuffer` for efficient data transfer and processing, and notes the option for zero-copy buffer usage.

```javascript
const dtype = 'float32';
const device = tf.backend().device;
const aData = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
const bData = [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4];
const expected = [2, 4, 6, 8, 6, 8, 10, 12, 10, 12, 14, 16, 14, 16, 18, 20];
const aBuffer = createGPUBufferFromData(device, aData, dtype);
const shape = [aData.length];
// To use zeroCopy, use {buffer: aBuffer, zeroCopy: true} instead and destroy
// aBuffer untill all access is done.
const a = tf.tensor({buffer: aBuffer}, shape, dtype);
const b = tf.tensor(bData, shape, dtype);
const result = tf.add(a, b);
result.print();
a.dispose();
b.dispose();
result.dispose();
aBuffer.destroy();
await tf.setBackend(savedBackend);
```

--------------------------------

### Create and Apply SimpleRNN Layer in JavaScript

Source: https://js.tensorflow.org/api/latest/index

This snippet demonstrates how to create a `SimpleRNN` layer with specified units and `returnSequences` set to true. It then applies this layer to an input tensor and logs the shape of the output. The input tensor has a shape of [10, 20], representing 10 time steps with 20 features each. The output shape depends on the layer's configuration, including the number of units and the `returnSequences` flag.

```javascript
const rnn = tf.layers.simpleRNN({
  units: 8,
  returnSequences: true
});

// Create an input with 10 time steps.
const input = tf.input({shape: [10, 20]});
const output = rnn.apply(input);

console.log(JSON.stringify(output.shape));
// [null, 10, 8]: 1st dimension is unknown batch size; 2nd dimension is the
// same as the sequence length of `input`, due to `returnSequences`: `true`;
// 3rd dimension is the `SimpleRNNCell`'s number of units.
```

--------------------------------

### tf.GraphModel Methods

Source: https://js.tensorflow.org/api/latest/index

Documentation for the methods available on the tf.GraphModel class, including saving, loading, and prediction functionalities.

```APIDOC
## tf.GraphModel.loadSync

### Description
Synchronously constructs the in-memory weight map and compiles the inference graph.

### Method
`loadSync`

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
None

### Request Example
```json
{
  "artifacts": "io.ModelArtifacts"
}
```

### Response
#### Success Response (200)
- **boolean** - Indicates if the loading was successful.

#### Response Example
```json
{
  "example": true
}
```

## tf.GraphModel.save

### Description
Saves the configuration and/or weights of the GraphModel. An `IOHandler` manages the storing or transmission of serialized data.

### Method
`save`

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **handlerOrURL** (io.IOHandler|string) - An instance of `IOHandler` or a URL-like string shortcut.
- **config** (Object) - Options for saving the model. Optional.
  - **trainableOnly** (boolean) - Whether to save only the trainable weights.
  - **includeOptimizer** (boolean) - Whether to save the optimizer, if it exists. Defaults to `false`.

### Request Example
```json
{
  "handlerOrURL": "localstorage://my-model-1",
  "config": {
    "trainableOnly": false,
    "includeOptimizer": true
  }
}
```

### Response
#### Success Response (200)
- **io.SaveResult** - An object containing information about the saved artifacts.

#### Response Example
```json
{
  "example": {
    "modelArtifactsInfo": {
      "dateSaved": "2023-10-27T10:00:00Z",
      "modelTopologyType": "JSON",
      "weightSpecs": [
        {
          "name": "dense_1/kernel",
          "shape": [784, 10],
          "dtype": "float32"
        }
      ],
      "weightDataBytes": 7840
    }
  }
}
```

## tf.GraphModel.predict

### Description
Executes inference for the input tensors.

### Method
`predict`

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **inputs** (tf.Tensor|tf.Tensor[]|{[name: string]: tf.Tensor}) - The input tensors for the model.
- **config** (Object) - Prediction configuration. Optional.
  - **batchSize** (number) - Batch size. Defaults to 32.
  - **verbose** (boolean) - Verbosity mode. Defaults to false.

### Request Example
```json
{
  "inputs": [
    {
      "dtype": "float32",
      "shape": [1, 28, 28, 1],
      "data": [0.0, 0.1, ...]
    }
  ],
  "config": {
    "batchSize": 32,
    "verbose": true
  }
}
```

### Response
#### Success Response (200)
- **tf.Tensor|tf.Tensor[]|{[name: string]: tf.Tensor}** - The output tensor(s) from the model.

#### Response Example
```json
{
  "example": [
    {
      "dtype": "float32",
      "shape": [1, 10],
      "data": [0.1, 0.05, ...]
    }
  ]
}
```

## tf.GraphModel.predictAsync

### Description
Executes inference for the input tensors in async fashion. Use this method when your model contains control flow ops.

### Method
`predictAsync`

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **inputs** (tf.Tensor|tf.Tensor[]|{[name: string]: tf.Tensor}) - The input tensors for the model.
- **config** (Object) - Prediction configuration. Optional.
  - **batchSize** (number) - Batch size. Defaults to 32.
  - **verbose** (boolean) - Verbosity mode. Defaults to false.

### Request Example
```json
{
  "inputs": [
    {
      "dtype": "float32",
      "shape": [1, 28, 28, 1],
      "data": [0.0, 0.1, ...]
    }
  ],
  "config": {
    "batchSize": 32,
    "verbose": true
  }
}
```

### Response
#### Success Response (200)
- **Promise<tf.Tensor|tf.Tensor[]|{[name: string]: tf.Tensor}>** - A promise that resolves to the output tensor(s) from the model.

#### Response Example
```json
{
  "example": [
    {
      "dtype": "float32",
      "shape": [1, 10],
      "data": [0.1, 0.05, ...]
    }
  ]
}
```

## tf.GraphModel.execute

### Description
Executes inference for the model for given input tensors.

### Method
`execute`

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **inputs** (tf.Tensor|tf.Tensor[]|{[name: string]: tf.Tensor}) - Tensor, tensor array or tensor map of the inputs for the model, keyed by the input node names.
- **outputs** (string|string[]) - Output node name from the TensorFlow model. If no outputs are specified, the default outputs of the model would be used. You can inspect intermediate nodes of the model by adding them to the outputs array. Optional.

### Request Example
```json
{
  "inputs": [
    {
      "dtype": "float32",
      "shape": [1, 28, 28, 1],
      "data": [0.0, 0.1, ...]
    }
  ],
  "outputs": ["output_node_name"]
}
```

### Response
#### Success Response (200)
- **tf.Tensor|tf.Tensor[]** - The output tensor(s) from the model.

#### Response Example
```json
{
  "example": [
    {
      "dtype": "float32",
      "shape": [1, 10],
      "data": [0.1, 0.05, ...]
    }
  ]
}
```

## tf.GraphModel.executeAsync

### Description
Executes inference for the model for given input tensors in async fashion. Use this method when your model contains control flow ops.

### Method
`executeAsync`

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **inputs** (tf.Tensor|tf.Tensor[]|{[name: string]: tf.Tensor}) - Tensor, tensor array or tensor map of the inputs for the model, keyed by the input node names.
- **outputs** (string|string[]) - Output node name from the TensorFlow model. If no outputs are specified, the default outputs of the model would be used. You can inspect intermediate nodes of the model by adding them to the outputs array. Optional.

### Request Example
```json
{
  "inputs": [
    {
      "dtype": "float32",
      "shape": [1, 28, 28, 1],
      "data": [0.0, 0.1, ...]
    }
  ],
  "outputs": ["output_node_name"]
}
```

### Response
#### Success Response (200)
- **Promise<tf.Tensor|tf.Tensor[]>** - A promise that resolves to the output tensor(s) from the model.

#### Response Example
```json
{
  "example": [
    {
      "dtype": "float32",
      "shape": [1, 10],
      "data": [0.1, 0.05, ...]
    }
  ]
}
```
```

--------------------------------

### tf.print

Source: https://js.tensorflow.org/api/latest/index

Prints information about a tf.Tensor, optionally including verbose details.

```APIDOC
## POST /tf.print

### Description
Prints information about the tf.Tensor including its data.

### Method
POST

### Endpoint
/tf.print

### Parameters
#### Query Parameters
- **x** (tf.Tensor|TypedArray|Array) - Required - The tensor to be printed.
- **verbose** (boolean) - Optional - Whether to print verbose information about the ` Tensor`, including dtype and size.

### Request Example
```json
{
  "x": [1, 2, 3, 4],
  "verbose": true
}
```

### Response
#### Success Response (200)
- **void** - This function does not return a value.

#### Response Example
```json
// No response body for this operation
```
```

--------------------------------

### Fit Quadratic Function with tf.SGDOptimizer in JavaScript

Source: https://js.tensorflow.org/api/latest/index

This snippet demonstrates how to fit a quadratic function using tf.SGDOptimizer in JavaScript. It defines input data (xs, ys), a quadratic model (f), a loss function (loss), and then iteratively trains the model using the optimizer. Finally, it logs the learned coefficients and predictions.

```javascript
// Fit a quadratic function by learning the coefficients a, b, c.
const xs = tf.tensor1d([0, 1, 2, 3]);
const ys = tf.tensor1d([1.1, 5.9, 16.8, 33.9]);

const a = tf.scalar(Math.random()).variable();
const b = tf.scalar(Math.random()).variable();
const c = tf.scalar(Math.random()).variable();

// y = a * x^2 + b * x + c.
const f = x => a.mul(x.square()).add(b.mul(x)).add(c);
const loss = (pred, label) => pred.sub(label).square().mean();

const learningRate = 0.01;
const optimizer = tf.train.sgd(learningRate);

// Train the model.
for (let i = 0; i < 10; i++) {
   optimizer.minimize(() => loss(f(xs), ys));
}

// Make predictions.
console.log(
     `a: ${a.dataSync()}, b: ${b.dataSync()}, c: ${c.dataSync()}`);
const preds = f(xs).dataSync();
preds.forEach((pred, i) => {
   console.log(`x: ${i}, pred: ${pred}`);
});

```

--------------------------------

### Model Training Configuration

Source: https://js.tensorflow.org/api/latest/index

Configuration options for training a TensorFlow.js model, including verbosity, callbacks, validation data, and epoch settings.

```APIDOC
## Model Training Parameters

### Description
Configuration options for training a TensorFlow.js model, including verbosity, callbacks, validation data, and epoch settings.

### Parameters

#### Training Options

- **`verbose`** (number) - Optional - Verbosity level. Expected to be 0, 1, or 2. Default: 1.
  - 0: No printed message during `fit()` call.
  - 1: In Node.js, prints progress bar, loss/metric updates, and training speed. In the browser, no action. (Default)
  - 2: Not implemented yet.

- **`callbacks`** (Array<Callback> | Callback | CustomCallbackArgs) - Optional - List of callbacks to be called during training. Supported callbacks include:
  - `onTrainBegin(logs)`: Called when training starts.
  - `onTrainEnd(logs)`: Called when training ends.
  - `onEpochBegin(epoch, logs)`: Called at the start of every epoch.
  - `onEpochEnd(epoch, logs)`: Called at the end of every epoch.
  - `onBatchBegin(batch, logs)`: Called at the start of every batch.
  - `onBatchEnd(batch, logs)`: Called at the end of every batch.
  - `onYield(epoch, batch, logs)`: Called every `yieldEvery` milliseconds.

- **`validationData`** (Array<Tensor | Tensor[]> | Dataset | Array<Tensor | Tensor[] | Tensor[]>) - Optional - Data on which to evaluate the loss and metrics at the end of each epoch. The model will not be trained on this data. Can be:
  - `[xVal, yVal]` (Tensors, Array of Tensors, or map of string to Tensor).
  - `[xVal, yVal, valSampleWeights]` (Not implemented yet).
  - A `Dataset` object with elements of the form `{xs: xVal, ys: yVal}`.

- **`validationBatchSize`** (number) - Optional - Batch size for validation. Used only if `validationData` is an array of Tensors. Defaults to 32.

- **`validationBatches`** (number) - Optional - Number of batches to draw from `validationData` (if it's a Dataset) before stopping validation each epoch.

- **`yieldEvery`** (YieldEveryOptions | 'auto' | 'batch' | 'epoch' | number | 'never') - Optional - Configures the frequency of yielding the main thread to other tasks. Defaults to 'auto'.
  - `'auto'`: Yields at a certain frame rate (currently 125ms).
  - `'batch'`: Yield every batch.
  - `'epoch'`: Yield every epoch.
  - `number`: Yield every `number` milliseconds.
  - `'never'`: Never yield automatically.

- **`initialEpoch`** (number) - Optional - Epoch at which to start training. Useful for resuming training. Defaults to 0.
```

--------------------------------

### RNN Layer with GRUCell

Source: https://js.tensorflow.org/api/latest/index

Illustrates how to construct an RNN layer using multiple GRUCells, suitable for processing sequences of data.

```APIDOC
## RNN Layer with GRUCell

### Description
Instance(s) of `GRUCell` can be used to construct `RNN` layers. A common workflow is to combine cells into a `StackedRNNCell` (internally) and use it to create an RNN. The `tf.layers.rnn` function facilitates this.

### Method
`tf.layers.rnn(args)`

### Endpoint
N/A (Client-side JavaScript API)

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **args** (Object) Configuration object for the RNN layer.
  - **cell** (Array<Layer> | Layer) One or more GRUCell instances.
  - **returnSequences** (boolean) Whether to return the full sequence of outputs.

### Request Example
```javascript
const cells = [
   tf.layers.gruCell({units: 4}),
   tf.layers.gruCell({units: 8}),
];
const rnn = tf.layers.rnn({cell: cells, returnSequences: true});

// Create an input with 10 time steps and a length-20 vector at each step.
const input = tf.input({shape: [10, 20]});
const output = rnn.apply(input);

console.log(JSON.stringify(output.shape));
// [null, 10, 8]: 1st dimension is unknown batch size; 2nd dimension is the
// same as the sequence length of `input`, due to `returnSequences`: `true`;
// 3rd dimension is the last `gruCell`'s number of units.
```

### Response
#### Success Response (200)
- **output** (Tensor) The output tensor of the RNN layer.

#### Response Example
```json
[null, 10, 8]
```
```

--------------------------------

### GRUCell Usage

Source: https://js.tensorflow.org/api/latest/index

Demonstrates how to use the GRUCell layer, which processes input for a single time step.

```APIDOC
## GRUCell

### Description
`GRUCell` is distinct from the `RNN` subclass `GRU` in that its `apply` method takes the input data of only a single time step and returns the cell's output at the time step.

### Method
`apply(inputs, states)`

### Endpoint
N/A (Client-side JavaScript API)

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **inputs** (Tensor) Input tensor for a single time step.
- **states** (Tensor) State tensor(s) for the cell.

### Request Example
```javascript
const cell = tf.layers.gruCell({units: 2});
const input = tf.input({shape: [10]});
const output = cell.apply(input);

console.log(JSON.stringify(output.shape));
// [null, 10]: This is the cell's output at a single time step. The 1st
// dimension is the unknown batch size.
```

### Response
#### Success Response (200)
- **output** (Tensor) The cell's output at the single time step.

#### Response Example
```json
[null, 10]
```
```

--------------------------------

### Apply Layer with Concrete Tensors

Source: https://js.tensorflow.org/api/latest/index

Demonstrates how to use the `apply` method of a `tf.layers.Layer` with concrete `tf.Tensor` inputs. This executes the layer's computation directly. It shows a `tf.layers.dense` layer with specific initializers and bias settings, and how its output is affected by zero-initialized kernels and the absence of bias.

```javascript
const denseLayer = tf.layers.dense({
   units: 1,
   kernelInitializer: 'zeros',
   useBias: false
});

// Invoke the layer's apply() method with a [tf.Tensor](#class:Tensor) (with concrete
// numeric values).
const input = tf.ones([2, 2]);
const output = denseLayer.apply(input);

// The output's value is expected to be [[0], [0]], due to the fact that
// the dense layer has a kernel initialized to all-zeros and does not have
// a bias.
output.print();

```

--------------------------------

### tf.initializers.glorotUniform

Source: https://js.tensorflow.org/api/latest/index

The Glorot uniform initializer, also known as the Xavier uniform initializer. It draws samples from a uniform distribution within a calculated limit based on input and output units.

```APIDOC
## tf.initializers.glorotUniform (args)

### Description
Glorot uniform initializer, also called Xavier uniform initializer. It draws samples from a uniform distribution within [-limit, limit] where `limit` is `sqrt(6 / (fan_in + fan_out))` where `fan_in` is the number of input units in the weight tensor and `fan_out` is the number of output units in the weight tensor.

### Method
Function

### Endpoint
N/A (Client-side JavaScript function)

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
* **args** (Object) - Configuration object for the initializer.
* **seed** (number) - Optional. Random number generator seed.

### Request Example
```json
{
  "args": {},
  "seed": 123
}
```

### Response
#### Success Response (tf.initializers.Initializer)
Returns an instance of tf.initializers.Initializer.

#### Response Example
```json
{
  "instance": "tf.initializers.Initializer"
}
```

### Reference
Glorot & Bengio, AISTATS 2010 http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
```

--------------------------------

### tf.LayersModel.summary

Source: https://js.tensorflow.org/api/latest/index

Prints a text summary of the model's layers, including layer details, output shapes, parameter counts, and total trainable/non-trainable parameters.

```APIDOC
## GET /websites/js_tensorflow_api/layersModel/summary

### Description
Prints a text summary of the model's layers. The summary includes the name and type of all layers, their output shapes, the number of weight parameters, and the total number of trainable and non-trainable parameters.

### Method
GET

### Endpoint
/websites/js_tensorflow_api/layersModel/summary

### Parameters
#### Query Parameters
- **lineLength** (number) - Optional - Custom line length, in number of characters.
- **positions** (number[]) - Optional - Custom widths of each of the columns, as either fractions of `lineLength` or absolute number of characters.
- **printFn** ((message?: tf.any(), ...optionalParams: tf.any()[]) => void) - Optional - Custom print function to replace the default `console.log`.

### Request Example
```javascript
// Assuming 'model' is an instance of tf.LayersModel
model.summary();

// With custom options
model.summary(80, [30, 50, 65], console.log);
```

### Response
#### Success Response (200)
- **void** - This method does not return a value, it prints to the console or a provided function.
```

--------------------------------

### Apply GRUCell to a Single Time Step Input - JavaScript

Source: https://js.tensorflow.org/api/latest/index

Demonstrates applying a `GRUCell` to input data from a single time step. The `apply` method of `GRUCell` takes a single time step's input and returns the cell's output for that step. The output shape is `[null, 10]`, where `null` represents the unknown batch size.

```javascript
const cell = tf.layers.gruCell({units: 2});
const input = tf.input({shape: [10]});
const output = cell.apply(input);

console.log(JSON.stringify(output.shape));
// [null, 10]: This is the cell's output at a single time step. The 1st
// dimension is the unknown batch size.
```

--------------------------------

### tf.LayersModel.compile

Source: https://js.tensorflow.org/api/latest/index

Configures and prepares the model for training and evaluation by specifying the optimizer, loss function, and metrics.

```APIDOC
## POST /websites/js_tensorflow_api/layersModel/compile

### Description
Configures and prepares the model for training and evaluation. Compiling outfits the model with an optimizer, loss, and/or metrics. Calling `fit` or `evaluate` on an un-compiled model will throw an error.

### Method
POST

### Endpoint
/websites/js_tensorflow_api/layersModel/compile

### Parameters
#### Request Body
- **args** (Object) - Required - A `ModelCompileArgs` object specifying the loss, optimizer, and metrics.
  - **optimizer** (string|tf.train.Optimizer) - Required - An instance of `tf.train.Optimizer` or a string name for an Optimizer.
  - **loss** (string|string[]|{[outputName: string]: string}|LossOrMetricFn| LossOrMetricFn[]|{[outputName: string]: LossOrMetricFn}) - Required - Object function(s) or name(s) of object function(s). Can be a single loss, an array of losses, or a dictionary for multi-output models.
  - **metrics** (string|LossOrMetricFn|Array| {[outputName: string]: string | LossOrMetricFn}) - Optional - List of metrics to be evaluated by the model during training and testing. E.g., `['accuracy']`.

### Request Example
```javascript
const model = tf.sequential({layers: [tf.layers.dense({units: 1, inputShape: [10]})]});
model.compile({
  optimizer: 'sgd',
  loss: 'meanSquaredError',
  metrics: ['accuracy']
});
```

### Response
#### Success Response (200)
- **void** - This method does not return a value.
```

--------------------------------

### Take Initial Elements from Dataset with JavaScript

Source: https://js.tensorflow.org/api/latest/index

The `take` method creates a new dataset containing at most a specified number of initial elements from the original dataset. If the count is undefined, negative, or greater than the dataset size, all elements are included.

```javascript
const a = tf.data.array([1, 2, 3, 4, 5, 6]).take(3);
await a.forEachAsync(e => console.log(e));

```

--------------------------------

### Fetch Resources with tf.fetch

Source: https://js.tensorflow.org/api/latest/index

Provides a platform-specific implementation of the `fetch` API. It falls back to a global `fetch` if available, otherwise uses a built-in solution. Returns a Promise that resolves to a Response object.

```javascript
const resource = await tf.util.fetch('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs');
// handle response
```

--------------------------------

### LSTMCell Apply Method - TensorFlow.js

Source: https://js.tensorflow.org/api/latest/index

Demonstrates the application of an LSTMCell to a single time step's input. It shows how to initialize an LSTMCell and apply it to an input tensor, returning the cell's output and state for that time step. The output shape reflects the number of units and the unknown batch size.

```javascript
const cell = tf.layers.lstmCell({units: 2});
const input = tf.input({shape: [10]});
const output = cell.apply(input);

console.log(JSON.stringify(output.shape));
// [null, 10]: This is the cell's output at a single time step. The 1st
// dimension is the unknown batch size.
```

--------------------------------

### tf.getBackend

Source: https://js.tensorflow.org/api/latest/index

Returns the current backend name (cpu, webgl, etc).

```APIDOC
## tf.getBackend ()

### Description
Returns the current backend name (cpu, webgl, etc). The backend is responsible for creating tensors and executing operations on those tensors.

### Method
N/A (Backend function)

### Endpoint
N/A

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
None

### Request Example
```javascript
const backendName = tf.getBackend();
console.log(`The current backend is: ${backendName}`);
```

### Response
#### Success Response (N/A)
- **string** - The name of the current backend (e.g., 'cpu', 'webgl').

#### Response Example
```json
"webgl"
```
```

--------------------------------

### tf.layers.activation

Source: https://js.tensorflow.org/api/latest/index

Creates an Activation layer.

```APIDOC
## tf.layers.activation

### Description
Creates an Activation layer.

### Method
`tf.layers.activation(args)`

### Endpoint
N/A (Client-side API)

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
None

* **args** (Object) - Configuration options for the layer.
  * **activation** ('elu'|'hardSigmoid'|'linear'|'relu'|'relu6'| 'selu'|'sigmoid'|'softmax'|'softplus'|'softsign'|'tanh'|'swish'|'mish'|'gelu'|'gelu_new') - Name of the activation function to use.
  * **inputShape** ((null | number)[]) - If defined, will be used to create an input layer to insert before this layer. Only applicable to input layers.
  * **batchInputShape** ((null | number)[]) - If defined, will be used to create an input layer to insert before this layer. Only applicable to input layers.
  * **batchSize** (number) - Used to construct `batchInputShape` if `inputShape` is specified and `batchInputShape` is not.
  * **dtype** ('float32'|'int32'|'bool'|'complex64'|'string') - The data-type for this layer. Defaults to 'float32'. Only applicable to input layers.
  * **name** (string) - Name for this layer.
  * **trainable** (boolean) - Whether the weights of this layer are updatable by `fit`. Defaults to true.
  * **weights** (tf.Tensor[]) - Initial weight values of the layer.
  * **inputDType** ('float32'|'int32'|'bool'|'complex64'|'string') - Legacy support. Do not use for new code.

### Request Example
```json
{
  "activation": "relu",
  "inputShape": [5]
}
```

### Response
#### Success Response (200)
Activation layer instance.

#### Response Example
```json
{
  "activationLayer": "<tf.layers.Layer object>"
}
```
```

--------------------------------

### Save and Load GraphModel to Local Storage

Source: https://js.tensorflow.org/api/latest/index

Demonstrates how to save a tf.GraphModel's topology and weights to browser local storage and then load it back. This involves using `tf.loadGraphModel` to load the model, `model.save` with a 'localstorage://' URI, and then `tf.loadGraphModel` again to retrieve the saved model.

```javascript
const modelUrl =
    'https://storage.googleapis.com/tfjs-models/savedmodel/mobilenet_v2_1.0_224/model.json';
const model = await tf.loadGraphModel(modelUrl);
const zeros = tf.zeros([1, 224, 224, 3]);
model.predict(zeros).print();

const saveResults = await model.save('localstorage://my-model-1');

const loadedModel = await tf.loadGraphModel('localstorage://my-model-1');
console.log('Prediction from loaded model:');
model.predict(zeros).print();

```

--------------------------------

### tf.signal.frame

Source: https://js.tensorflow.org/api/latest/index

Expands input into frames of frameLength. Slides a window size with frameStep.

```APIDOC
## POST /tf.signal.frame

### Description
Expands input into frames of frameLength. Slides a window size with frameStep.

### Method
POST

### Endpoint
/tf.signal.frame

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **signal** (tf.Tensor1D) - Required - The input tensor to be expanded.
- **frameLength** (number) - Required - Length of each frame.
- **frameStep** (number) - Required - The frame hop size in samples.
- **padEnd** (boolean) - Optional - Whether to pad the end of signal with padValue.
- **padValue** (number) - Optional - A number to use where the input signal does not exist when padEnd is True.

### Request Example
```json
{
  "signal": [1, 2, 3],
  "frameLength": 2,
  "frameStep": 1
}
```

### Response
#### Success Response (200)
- **tensor** (tf.Tensor) - The framed tensor.

#### Response Example
```json
{
  "tensor": "[Tensor object representation]"
}
```
```

--------------------------------

### tf.initializers.leCunUniform

Source: https://js.tensorflow.org/api/latest/index

The LeCun uniform initializer. It draws samples from a uniform distribution within a calculated limit based on the number of input units.

```APIDOC
## tf.initializers.leCunUniform (args)

### Description
LeCun uniform initializer. It draws samples from a uniform distribution in the interval `[-limit, limit]` with `limit = sqrt(3 / fanIn)`, where `fanIn` is the number of input units in the weight tensor.

### Method
Function

### Endpoint
N/A (Client-side JavaScript function)

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
* **args** (Object) - Configuration object for the initializer.
* **seed** (number) - Optional. Random number generator seed.

### Request Example
```json
{
  "args": {},
  "seed": 102
}
```

### Response
#### Success Response (tf.initializers.Initializer)
Returns an instance of tf.initializers.Initializer.

#### Response Example
```json
{
  "instance": "tf.initializers.Initializer"
}
```
```

--------------------------------

### Save and Load Layers Model to IndexedDB

Source: https://js.tensorflow.org/api/latest/index

Illustrates saving a sequential model's topology and weights to the browser's IndexedDB and subsequently loading it. IndexedDB offers more storage capacity than local storage, making it suitable for larger models.

```javascript
const model = tf.sequential(
     {layers: [tf.layers.dense({units: 1, inputShape: [3]})}]});
console.log('Prediction from original model:');
model.predict(tf.ones([1, 3])).print();

const saveResults = await model.save('indexeddb://my-model-1');

const loadedModel = await tf.loadLayersModel('indexeddb://my-model-1');
console.log('Prediction from loaded model:');
loadedModel.predict(tf.ones([1, 3])).print();

```

--------------------------------

### Browser API

Source: https://js.tensorflow.org/api/latest/index

APIs for interacting with browser-specific functionalities, such as drawing tensors to a canvas and creating tensors from images.

```APIDOC
## tf.browser.draw

### Description
Draws a `tf.Tensor` to an HTML canvas element. Handles different input tensor dtypes and shapes for grayscale, RGB, and RGBA drawing.

### Method
`tf.browser.draw(image, canvas, options?)`

### Parameters
#### Path Parameters
- **image** (tf.Tensor2D|tf.Tensor3D|TypedArray|Array) - The tensor to draw. Supported shapes include `[height, width]`, `[height, width, 1]` (grayscale), `[height, width, 3]` (RGB), and `[height, width, 4]` (RGBA).
- **canvas** (HTMLCanvasElement) - The canvas element to draw the tensor onto.
- **options** (Object) - Optional configuration object.
  - **imageOptions** (ImageOptions) - Options to customize the image tensor values.
  - **contextOptions** (ContextOptions) - Options to configure the canvas context.

### Returns
void

## tf.browser.fromPixels

### Description
Creates a `tf.Tensor` from various image formats (e.g., `ImageData`, `HTMLImageElement`). Supports specifying the number of output channels.

### Method
`tf.browser.fromPixels(pixels, numChannels?)`

### Parameters
#### Path Parameters
- **pixels** (PixelData|ImageData|HTMLImageElement|HTMLCanvasElement| HTMLVideoElement|ImageBitmap) - The input image data. Must be 4-channel. Can also be an object with `data`, `width`, and `height` attributes.
- **numChannels** (number) - Optional. The number of channels for the output tensor. Defaults to 3 (ignores alpha channel).

### Returns
tf.Tensor3D

### Request Example
```json
{
  "pixels": "ImageData object",
  "numChannels": 3
}
```

## tf.browser.fromPixelsAsync

### Description
Asynchronously creates a `tf.Tensor` from various image formats. It prioritizes using `ImageBitmap` if the `WRAP_TO_IMAGEBITMAP` flag is enabled.

### Method
`tf.browser.fromPixelsAsync(pixels, numChannels?)`

### Parameters
#### Path Parameters
- **pixels** (PixelData|ImageData|HTMLImageElement|HTMLCanvasElement| HTMLVideoElement|ImageBitmap) - The input image data. Must be 4-channel. Can also be an object with `data`, `width`, and `height` attributes.
- **numChannels** (number) - Optional. The number of channels for the output tensor. Defaults to 3 (ignores alpha channel).

### Returns
Promise<tf.Tensor>

### Request Example
```json
{
  "pixels": "ImageData object",
  "numChannels": 4
}
```

## tf.browser.toPixels

### Description
Draws a `tf.Tensor` of pixel values to a `Uint8ClampedArray` or optionally to an HTML canvas. Handles 'float32' tensors (range [0-1]) and 'int32' tensors (range [0-255]).

### Method
`tf.browser.toPixels(img, canvas?)`

### Parameters
#### Path Parameters
- **img** (tf.Tensor2D|tf.Tensor3D|TypedArray|Array) - The input tensor representing pixel data. Supported shapes include `[height, width]` (grayscale), `[height, width, 1]` (grayscale), `[height, width, 3]` (RGB), and `[height, width, 4]` (RGBA).
- **canvas** (HTMLCanvasElement) - Optional. The canvas element to draw the pixel data onto.

### Returns
Promise<Uint8ClampedArray> - A promise that resolves when the drawing is complete.
```

--------------------------------

### tf.Tensor Methods

Source: https://js.tensorflow.org/api/latest/index

Methods available on tf.Tensor objects, including buffer and array access.

```APIDOC
## POST /tf.Tensor/buffer

### Description
Returns a promise of tf.TensorBuffer that holds the underlying data.

### Method
POST

### Endpoint
/tf.Tensor/buffer

### Parameters
#### Path Parameters
- **tensorId** (string) - Required - The ID of the tensor.

### Request Example
```json
{
  "tensorId": "tensor_123"
}
```

### Response
#### Success Response (200)
- **Promise<tf.TensorBuffer>** - A promise resolving to the tensor buffer.

#### Response Example
```json
{
  "tensorBuffer": {"shape": [2, 2], "dtype": "float32", "values": [1.0, 2.0, 3.0, 4.0]}
}
```

## POST /tf.Tensor/bufferSync

### Description
Returns a tf.TensorBuffer that holds the underlying data synchronously.

### Method
POST

### Endpoint
/tf.Tensor/bufferSync

### Parameters
#### Path Parameters
- **tensorId** (string) - Required - The ID of the tensor.

### Request Example
```json
{
  "tensorId": "tensor_123"
}
```

### Response
#### Success Response (200)
- **tf.TensorBuffer** - The tensor buffer.

#### Response Example
```json
{
  "tensorBuffer": {"shape": [2, 2], "dtype": "float32", "values": [1.0, 2.0, 3.0, 4.0]}
}
```

## POST /tf.Tensor/array

### Description
Returns a promise of the tensor's data as a nested array.

### Method
POST

### Endpoint
/tf.Tensor/array

### Parameters
#### Path Parameters
- **tensorId** (string) - Required - The ID of the tensor.

### Request Example
```json
{
  "tensorId": "tensor_123"
}
```

### Response
#### Success Response (200)
- **Promise<Array>** - A promise resolving to the tensor's data as a nested array.

#### Response Example
```json
{
  "array": [[1.0, 2.0], [3.0, 4.0]]
}
```
```

--------------------------------

### Copy TensorFlow.js Model Between Storage Mediums

Source: https://js.tensorflow.org/api/latest/index

Illustrates copying a TensorFlow.js model from one URL to another, supporting both same-medium and cross-medium copies (e.g., Local Storage to IndexedDB). It includes saving a model, listing models, copying, listing again, and finally removing the models.

```javascript
// First create and save a model.
const model = tf.sequential();
model.add(tf.layers.dense(
     {units: 1, inputShape: [10], activation: 'sigmoid'}));
await model.save('localstorage://demo/management/model1');

// Then list existing models.
console.log(JSON.stringify(await tf.io.listModels()));

// Copy the model, from Local Storage to IndexedDB.
await tf.io.copyModel(
     'localstorage://demo/management/model1',
     'indexeddb://demo/management/model1');

// List models again.
console.log(JSON.stringify(await tf.io.listModels()));

// Remove both models.
await tf.io.removeModel('localstorage://demo/management/model1');
await tf.io.removeModel('indexeddb://demo/management/model1');

```

--------------------------------

### Tensor Creation API

Source: https://js.tensorflow.org/api/latest/index

This section covers the `tf.tensor` function for creating tensors from various data sources and shapes.

```APIDOC
## POST /api/tensors

### Description
Creates a tf.Tensor with the provided values, shape, and dtype. This is the primary method for tensor instantiation in TensorFlow.js.

### Method
POST

### Endpoint
/api/tensors

### Parameters
#### Query Parameters
- **values** (Array | HTMLCanvasElement | ImageData | HTMLVideoElement | WebGLTexture | GPUBuffer) - Required - The data to populate the tensor.
- **shape** (number[]) - Optional - The shape of the tensor. If not provided, it's inferred from the values.
- **dtype** (string) - Optional - The data type of the tensor (e.g., 'float32', 'int32', 'bool'). Defaults to 'float32'.

### Request Body
(Not applicable for this function, parameters are passed directly)

### Request Example
```javascript
// Create a 1D tensor (vector)
tf.tensor([1, 2, 3, 4]).print();

// Create a 2D tensor (matrix)
tf.tensor([[1, 2], [3, 4]]).print();

// Create a tensor with a specified shape
tf.tensor([1, 2, 3, 4], [2, 2]).print();

// Create a tensor from a WebGL texture (example for WebGL2)
// Assuming 'texture', 'width', 'height', 'channels' are defined based on WebGL context
// const a = tf.tensor({ texture, height, width, channels: 'BR' }, [height * width * 2]);
// a.print();

// Create a tensor from a WebGPU buffer (example for WebGPU)
// Assuming 'gpuBuffer', 'logicalShape', 'dtype' are defined based on WebGPU context
// const b = tf.tensor({ data: gpuBuffer, shape: logicalShape, dtype: 'float32', zeroCopy: true });
// b.print();
```

### Response
#### Success Response (200)
- **Tensor** (tf.Tensor) - The newly created tensor object.

#### Response Example
```json
{
  "tensor": "tf.Tensor object"
}
```
```

--------------------------------

### tf.signal.hammingWindow

Source: https://js.tensorflow.org/api/latest/index

Generate a hamming window.

```APIDOC
## POST /tf.signal.hammingWindow

### Description
Generate a hamming window.
See: https://en.wikipedia.org/wiki/Window_function#Hann_and_Hamming_windows

### Method
POST

### Endpoint
/tf.signal.hammingWindow

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **windowLength** (number) - Required - The length of the window.

### Request Example
```json
{
  "windowLength": 10
}
```

### Response
#### Success Response (200)
- **tensor** (tf.Tensor1D) - The generated Hamming window tensor.

#### Response Example
```json
{
  "tensor": "[Tensor object representation]"
}
```
```

--------------------------------

### tf.fetch

Source: https://js.tensorflow.org/api/latest/index

Returns a platform-specific implementation of `fetch`.

```APIDOC
## tf.fetch (path, requestInits?)

### Description
Returns a platform-specific implementation of `fetch`. If `fetch` is defined on the global object, `tf.util.fetch` returns that function. Otherwise, it returns a platform-specific solution.

### Method
N/A (Utility function)

### Endpoint
N/A

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
None

### Request Example
```javascript
const resource = await tf.util.fetch('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs');
// handle response
```

### Response
#### Success Response (N/A)
- **Promise<Response>** - A promise that resolves with the Response object.

#### Response Example
```json
// The Response object from the fetch API
```
```

--------------------------------

### Glorot Normal Initializer for TensorFlow.js Layers

Source: https://js.tensorflow.org/api/latest/index

Initializes layer weights using the Glorot normal (Xavier normal) distribution. This initializer helps in maintaining variance of activations across layers, particularly useful for deep networks.

```javascript
const initializer = tf.initializers.glorotNormal ();
```

--------------------------------

### GRU Layer Parameters

Source: https://js.tensorflow.org/api/latest/index

Details the parameters available when creating a GRU layer directly using tf.layers.gru().

```APIDOC
## GRU Layer Parameters

### Description
Parameters for creating a `GRU` layer, which wraps a single GRU cell.

### Method
`tf.layers.gru(args)`

### Endpoint
N/A (Client-side JavaScript API)

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **args** (Object) Configuration object for the GRU layer.
  - **recurrentActivation** ('elu'|'hardSigmoid'|'linear'|'relu'|'relu6'| 'selu'|'sigmoid'|'softmax'|'softplus'|'softsign'|'tanh'|'swish'|'mish'|'gelu'|'gelu_new') Activation function for the recurrent step. Defaults to 'hardSigmoid'.
  - **implementation** (number) Implementation mode (1 or 2). Default behavior uses implementation 2 for performance.
  - **resetAfter** (boolean) Convention for reset gate application (only `false` is supported).
  - **units** (number) Dimensionality of the output space.
  - **activation** ('elu'|'hardSigmoid'|'linear'|'relu'|'relu6'| 'selu'|'sigmoid'|'softmax'|'softplus'|'softsign'|'tanh'|'swish'|'mish'|'gelu'|'gelu_new') Activation function. Defaults to 'tanh'.
  - **useBias** (boolean) Whether the layer uses a bias vector.
  - **kernelInitializer** (string|tf.initializers.Initializer) Initializer for the `kernel` weights matrix.
  - **recurrentInitializer** (string|tf.initializers.Initializer) Initializer for the `recurrentKernel` weights matrix.
  - **biasInitializer** (string|tf.initializers.Initializer) Initializer for the bias vector.
  - **kernelRegularizer** (string|Regularizer) Regularizer function for the `kernel` weights.
  - **recurrentRegularizer** (string|Regularizer) Regularizer function for the `recurrent_kernel` weights.
  - **biasRegularizer** (string|Regularizer) Regularizer function for the bias vector.
  - **kernelConstraint** (string|tf.constraints.Constraint) Constraint function for the `kernel` weights.
  - **recurrentConstraint** (string|tf.constraints.Constraint) Constraint function for the `recurrentKernel` weights.
  - **biasConstraint** (string|tf.constraints.Constraint) Constraint function for the bias vector.
  - **dropout** (number) Fraction of units to drop for input transformation.
  - **recurrentDropout** (number) Fraction of units to drop for recurrent state transformation.

### Request Example
```javascript
const gruLayer = tf.layers.gru({
  units: 32,
  activation: 'relu',
  recurrentActivation: 'hardSigmoid',
  dropout: 0.1,
  recurrentDropout: 0.1,
  returnSequences: true
});

const input = tf.input({shape: [10, 20]});
const output = gruLayer.apply(input);

console.log(JSON.stringify(output.shape));
// Example output shape: [null, 10, 32]
```

### Response
#### Success Response (200)
- **output** (Tensor) The output tensor of the GRU layer.

#### Response Example
```json
[null, 10, 32]
```
```

--------------------------------

### Image Transform API

Source: https://js.tensorflow.org/api/latest/index

API for applying transformations to images.

```APIDOC
## tf.image.transform

### Description
Applies the given transform(s) to the image(s).

### Method
Not specified (assumed to be a function call)

### Endpoint
N/A

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **image** (tf.Tensor4D|TypedArray|Array) - Required - 4d tensor of shape `[batch, imageHeight, imageWidth, depth]`.
- **transforms** (tf.Tensor2D|TypedArray|Array) - Required - Projective transform matrix/matrices. A tensor1d of length 8 or tensor of size N x 8. If one row of transforms is [a0, a1, a2, b0, b1, b2, c0, c1], then it maps the output point (x, y) to a transformed input point (x', y') = ((a0 x + a1 y + a2) / k, (b0 x + b1 y + b2) / k), where k = c0 x + c1 y + 1. The transforms are inverted compared to the transform mapping input points to output points.
- **interpolation** ('nearest'|'bilinear') - Optional - Interpolation mode. Supported values: 'nearest', 'bilinear'. Default to 'nearest'.
- **fillMode** ('constant'|'reflect'|'wrap'|'nearest') - Optional - Points outside the boundaries of the input are filled according to the given mode, one of 'constant', 'reflect', 'wrap', 'nearest'. Default to 'constant'.
  - 'reflect': (d c b a | a b c d | d c b a ) The input is extended by reflecting about the edge of the last pixel.
  - 'constant': (k k k k | a b c d | k k k k) The input is extended by filling all values beyond the edge with the same constant value k.
  - 'wrap': (a b c d | a b c d | a b c d) The input is extended by wrapping around to the opposite edge.
  - 'nearest': (a a a a | a b c d | d d d d) The input is extended by the nearest pixel.
- **fillValue** (number) - Optional - A float represents the value to be filled outside the boundaries when fillMode is 'constant'.
- **outputShape** ([number, number]) - Optional - The desired output shape of the transformed image.

### Request Example
```javascript
const transforms = tf.tensor2d([[1, 0, 0, 0, 1, 0, 0, 0]]); // Identity transform
tf.image.transform(imageTensor, transforms, 'bilinear', 'reflect', 0, [256, 256])
```

### Response
#### Success Response (200)
- **result** (tf.Tensor4D) - The transformed image tensor.
```

--------------------------------

### Profile Function Execution with tf.profile

Source: https://js.tensorflow.org/api/latest/index

Executes a function and returns a promise with information about memory usage, including newBytes, newTensors, peakBytes, and kernel details. This helps in analyzing performance bottlenecks.

```javascript
const profile = await tf.profile(() => {
   const x = tf.tensor1d([1, 2, 3]);
   let x2 = x.square();
   x2.dispose();
   x2 = x.square();
   x2.dispose();
   return x;
});

console.log(`newBytes: ${profile.newBytes}`);
console.log(`newTensors: ${profile.newTensors}`);
console.log(`byte usage over all kernels: ${profile.kernels.map(k =>
k.totalBytesSnapshot)}
`);

```

--------------------------------

### Load Layers Model from User Files

Source: https://js.tensorflow.org/api/latest/index

Loads a Layer model from files selected by the user via HTML input elements. This method requires corresponding HTML elements with IDs 'json-upload' and 'weights-upload' to be present in the DOM.

```javascript
// Note: this code snippet will not work without the HTML elements in the
//   page
const jsonUpload = document.getElementById('json-upload');
const weightsUpload = document.getElementById('weights-upload');

const model = await tf.loadLayersModel(
     tf.io.browserFiles([jsonUpload.files[0], weightsUpload.files[0]]));

```

--------------------------------

### Pad Tensor with tf.pad

Source: https://js.tensorflow.org/api/latest/index

Pads a tf.Tensor with a given value and paddings using CONSTANT mode. For REFLECT and SYMMETRIC modes, tf.mirrorPad should be used. Supports various backends like webgpu, webgl, wasm, and cpu.

```javascript
const x = tf.tensor1d([1, 2, 3, 4]);
x.pad([[1, 2]]).print();
```

--------------------------------

### tf.io.http

Source: https://js.tensorflow.org/api/latest/index

Creates an IOHandler that sends model artifacts to an HTTP server using a multipart/form-data POST request.

```APIDOC
## POST /upload (tf.io.http)

### Description
Creates an IOHandler subtype that sends model artifacts to an HTTP server. An HTTP request of the `multipart/form-data` mime type will be sent to the `path` URL. The form data includes artifacts that represent the topology and/or weights of the model.

### Method
POST (default) or PUT (configurable via `requestInit`)

### Endpoint
`path` (string): A URL path to the model. Can be an absolute HTTP path or a relative path.

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
None (handled internally by `tf.io.http`)

### Request Example
```javascript
const model = tf.sequential();
model.add(tf.layers.dense({units: 1, inputShape: [100], activation: 'sigmoid'}));

// Using default POST
const saveResult = await model.save('http://model-server:5000/upload');
console.log(saveResult);

// Using PUT with custom requestInit
const saveResultPut = await model.save(tf.io.http('http://model-server:5000/upload', {requestInit: {method: 'PUT'}}));
console.log(saveResultPut);
```

### Response
#### Success Response (200)
`IOHandler`: An IOHandler object.

#### Response Example
```json
// Example response structure if the server sends one back
{
  "status": "success",
  "message": "Model uploaded successfully"
}
```

### Notes
- A server-side implementation (like the Flask example on GitHub Gist) is required to receive and process the model artifacts.
```

--------------------------------

### Calculate Log Softmax with TensorFlow.js

Source: https://js.tensorflow.org/api/latest/index

Computes the log softmax of a tensor. Supports multiple backends including WebGPU, WebGL, WASM, and CPU. Accepts a tensor as input and returns a tensor.

```javascript
const a = tf.tensor2d([2, 4, 6, 1, 2, 3], [2, 3]);
a.logSoftmax().print();  // or tf.logSoftmax(a)
```

--------------------------------

### Load Layers Model from HTTP

Source: https://js.tensorflow.org/api/latest/index

Loads a Layer model from a given URL. This is a common scenario for loading pre-trained models hosted on web servers. It assumes the model's JSON configuration is accessible via HTTP.

```javascript
const model = await tf.loadLayersModel(
     'https://storage.googleapis.com/tfjs-models/tfjs/iris_v1/model.json');
model.summary();

```

--------------------------------

### Compute Log of One Plus Input Element-wise with tf.log1p

Source: https://js.tensorflow.org/api/latest/index

Computes the natural logarithm of the input tensor plus one element-wise (`ln(1 + x)`). It takes a tensor as input and returns a tensor with the computed values. Available for webgpu, webgl, wasm, and cpu backends.

```javascript
const x = tf.tensor1d([1, 2, Math.E - 1]);

x.log1p().print();  // or tf.log1p(x)
```

--------------------------------

### TensorFlow.js: Batch Dot Product using einsum

Source: https://js.tensorflow.org/api/latest/index

Computes the dot product for each corresponding row in two 2D tensors using `einsum`. The equation 'bi,bi->b' signifies that for each batch element 'b', the elements 'i' are multiplied and summed.

```javascript
const x = tf.tensor2d([[1, 2, 3], [4, 5, 6]]);
const y = tf.tensor2d([[0, 1, 2], [3, 4, 5]]);
x.print();
y.print();
tf.einsum('bi,bi->b', x, y).print();
```

--------------------------------

### tf.loadGraphModelSync

Source: https://js.tensorflow.org/api/latest/index

Loads a graph model synchronously using a provided IOHandler.

```APIDOC
## Load Graph Model Synchronously (tf.loadGraphModelSync)

### Description
Loads a graph model given a synchronous IO handler with a 'load' method. This function is useful when synchronous loading is required.

### Method
Synchronous (typically uses underlying synchronous IO operations)

### Endpoint
N/A (operates on `modelSource` argument)

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
None

### Parameters
- **modelSource** (io.IOHandlerSync | io.ModelArtifacts | [io.ModelJSON, ArrayBuffer]) - Required - The source from which to load the model. This can be:
    - An `io.IOHandlerSync` instance with a `load` method.
    - An `io.ModelArtifacts` object.
    - A tuple containing a model JSON object and an `ArrayBuffer` of weights.

### Request Example
```javascript
// Example assuming you have a synchronous IOHandler
const syncIOHandler = {
  load: async () => {
    // Synchronous loading logic here
    return {
      modelTopology: { ... }, // Your model topology
      weightSpecs: [],
      weightManifest: [],
      transferables: []
    };
  }
};
const graphModel = tf.loadGraphModelSync(syncIOHandler);
```

### Response
#### Success Response
- **tf.GraphModel**: A loaded TensorFlow.js GraphModel instance.

#### Response Example
```javascript
// A tf.GraphModel instance would be returned
console.log(graphModel);
```
```

--------------------------------

### Capture Spectrogram and Waveform from Microphone in TensorFlow.js

Source: https://js.tensorflow.org/api/latest/index

This function creates an iterator that captures audio from the device's microphone and generates frequency-domain spectrograms and time-domain waveforms. It requires user permission to access the microphone and is only available in browser environments. The captured data can be accessed as Tensors.

```javascript
const mic = await tf.data.microphone({
   fftSize: 1024,
   columnTruncateLength: 232,
   numFramesPerSpectrogram: 43,
   sampleRateHz:44100,
   includeSpectrogram: true,
   includeWaveform: true
});
const audioData = await mic.capture();
const spectrogramTensor = audioData.spectrogram;
spectrogramTensor.print();
const waveformTensor = audioData.waveform;
waveformTensor.print();
mic.stop();
```

--------------------------------

### Model Training: fit

Source: https://js.tensorflow.org/api/latest/index

Trains the model for a fixed number of epochs (iterations on a dataset). This method is used for the primary training loop of the model.

```APIDOC
## fit

### Description
Trains the model for a fixed number of epochs (iterations on a dataset).

### Method
`fit(x, y, args?)

### Parameters
#### Path Parameters
* x (tf.Tensor|tf.Tensor[]|{[inputName: string]: tf.Tensor}) - Required - tf.Tensor of training data, or an array of tf.Tensors if the model has multiple inputs. If all inputs in the model are named, you can also pass a dictionary mapping input names to tf.Tensors.
* y (tf.Tensor|tf.Tensor[]|{[inputName: string]: tf.Tensor}) - Required - tf.Tensor of target (label) data, or an array of tf.Tensors if the model has multiple outputs. If all outputs in the model are named, you can also pass a dictionary mapping output names to tf.Tensors.
* args (Object) - Optional - A `ModelFitArgs`, containing optional fields.
  * batchSize (number) - Optional - Number of samples per gradient update. If unspecified, it will default to 32.
  * epochs (number) - Optional - Integer number of times to iterate over the training data arrays.
  * verbose (ModelLoggingVerbosity | 2) - Optional - Verbosity level. Expected to be 0, 1, or 2. Default: 1.
    * 0 - No printed message during fit() call.
    * 1 - In Node.js (tfjs-node), prints the progress bar, together with real-time updates of loss and metric values and training speed. In the browser: no action. This is the default.
    * 2 - Not implemented yet.
  * callbacks (BaseCallback[]|CustomCallbackArgs|CustomCallbackArgs[]) - Optional - List of callbacks to be called during training. Can have one or more of the following callbacks:
    * `onTrainBegin(logs)`: called when training starts.
    * `onTrainEnd(logs)`: called when training ends.
    * `onEpochBegin(epoch, logs)`: called at the start of every epoch.
    * `onEpochEnd(epoch, logs)`: called at the end of every epoch.
    * `onBatchBegin(batch, logs)`: called at the start of every batch.
    * `onBatchEnd(batch, logs)`: called at the end of every batch.

### Request Example
```javascript
const model = tf.sequential({
     layers: [tf.layers.dense({units: 1, inputShape: [10]})]
});
model.compile({optimizer: 'sgd', loss: 'meanSquaredError'});
for (let i = 1; i < 5 ; ++i) {
   const h = await model.fit(tf.ones([8, 10]), tf.ones([8, 1]), {
       batchSize: 4,
       epochs: 3
   });
   console.log("Loss after Epoch " + i + " : " + h.history.loss[0]);
}
```

### Returns
Promise<tf.History>

### Notes
- Supported backends: webgpu, webgl, wasm, cpu.
```

--------------------------------

### Tile Tensor2D using TensorFlow.js

Source: https://js.tensorflow.org/api/latest/index

Constructs a 2-dimensional tensor by repeating it according to the `reps` parameter. The TensorFlow.js `tile` function allows for replication of tensor elements along specified dimensions. This operation is supported on webgpu, webgl, wasm, and cpu.

```javascript
const a = tf.tensor2d([1, 2, 3, 4], [2, 2]);

a.tile([1, 2]).print();  // or tf.tile(a, [1,2])

```

--------------------------------

### Construct tf.AdamaxOptimizer in JavaScript

Source: https://js.tensorflow.org/api/latest/index

Constructs a tf.AdamaxOptimizer that uses the Adamax algorithm. Adamax is a variant of the Adam optimization algorithm that improves its convergence properties.

```javascript
const learningRate = 0.002;
const optimizer = tf.train.adamax(learningRate);

```

--------------------------------

### tf.initializers.leCunNormal

Source: https://js.tensorflow.org/api/latest/index

The LeCun normal initializer. It draws samples from a truncated normal distribution centered on 0 with a standard deviation determined by the number of input units.

```APIDOC
## tf.initializers.leCunNormal (args)

### Description
LeCun normal initializer. It draws samples from a truncated normal distribution centered on 0 with `stddev = sqrt(1 / fanIn)` where `fanIn` is the number of input units in the weight tensor.

### Method
Function

### Endpoint
N/A (Client-side JavaScript function)

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
* **args** (Object) - Configuration object for the initializer.
* **seed** (number) - Optional. Random number generator seed.

### Request Example
```json
{
  "args": {},
  "seed": 101
}
```

### Response
#### Success Response (tf.initializers.Initializer)
Returns an instance of tf.initializers.Initializer.

#### Response Example
```json
{
  "instance": "tf.initializers.Initializer"
}
```

### Reference
Self-Normalizing Neural Networks Efficient Backprop
```

--------------------------------

### Constant Initializer for TensorFlow.js Layers

Source: https://js.tensorflow.org/api/latest/index

Initializes layer weights to a constant value. This is a simple initializer often used for bias terms or specific weight settings.

```javascript
const initializer = tf.initializers.constant ({value: 0.5});
```

--------------------------------

### Save and Load Layers Model to Local Storage

Source: https://js.tensorflow.org/api/latest/index

Demonstrates saving a sequential model's topology and weights to the browser's local storage and then loading it back. This is useful for persisting models client-side for quick access without network requests.

```javascript
const model = tf.sequential(
     {layers: [tf.layers.dense({units: 1, inputShape: [3]})}]});
console.log('Prediction from original model:');
model.predict(tf.ones([1, 3])).print();

const saveResults = await model.save('localstorage://my-model-1');

const loadedModel = await tf.loadLayersModel('localstorage://my-model-1');
console.log('Prediction from loaded model:');
loadedModel.predict(tf.ones([1, 3])).print();

```

--------------------------------

### Compute Softmax with TensorFlow.js

Source: https://js.tensorflow.org/api/latest/index

Computes the softmax normalized vector from logits. Supports multiple backends including WebGPU, WebGL, WASM, and CPU. Accepts logits and an optional dimension, returning a tensor.

```javascript
const a = tf.tensor1d([1, 2, 3]);
a.softmax().print();  // or tf.softmax(a)
```

```javascript
const a = tf.tensor2d([2, 4, 6, 1, 2, 3], [2, 3]);
a.softmax().print();  // or tf.softmax(a)
```

--------------------------------

### tf.meshgrid

Source: https://js.tensorflow.org/api/latest/index

Broadcasts parameters for evaluation on an N-D grid.

```APIDOC
## tf.meshgrid

### Description
Given N one-dimensional coordinate arrays `*args`, returns a list of N-D coordinate arrays for evaluating expressions on an N-D grid. Supports cartesian ('xy') and matrix ('ij') indexing conventions.

### Method
`tf.meshgrid(x?, y?, __2?)
`

### Parameters
- **x** (tf.Tensor | TypedArray | Array) - Tensor with rank >= 1. Optional.
- **y** (tf.Tensor | TypedArray | Array) - Tensor with rank >= 1. Optional.
- **__2** ({ indexing?: string; }) - Optional object for indexing configuration. Defaults to 'xy'.

### Returns
- **tf.Tensor[]** - A list of N-D coordinate arrays.

### Request Example
```javascript
const x = [1, 2, 3];
const y = [4, 5, 6];
const [X, Y] = tf.meshgrid(x, y);
// X = [[1, 2, 3],
//      [1, 2, 3],
//      [1, 2, 3]]
// Y = [[4, 4, 4],
//      [5, 5, 5],
//      [6, 6, 6]]
```
```

--------------------------------

### Metrics API

Source: https://js.tensorflow.org/api/latest/index

Provides metric functions for evaluating model performance.

```APIDOC
## tf.metrics.binaryAccuracy

### Description
Calculates the binary accuracy metric between true and predicted values. `yTrue` and `yPred` should contain values between 0 and 1.

### Method
`tf.metrics.binaryAccuracy(yTrue, yPred)`

### Parameters
#### Path Parameters
- **yTrue** (tf.Tensor) - The tensor containing the true labels.
- **yPred** (tf.Tensor) - The tensor containing the predicted labels.

### Request Example
```json
{
  "yTrue": "[[1, 1, 1, 1], [0, 0, 0, 0]]",
  "yPred": "[[1, 0, 1, 0], [0, 0, 0, 1]]"
}
```
```

--------------------------------

### Train on Batch

Source: https://js.tensorflow.org/api/latest/index

Executes a single gradient update on one batch of data. Returns only the loss and metric values.

```APIDOC
## POST /trainOnBatch

### Description
Runs a single gradient update on a single batch of data. This method returns only the loss and metric values, and does not support fine-grained options like verbosity or callbacks.

### Method
POST

### Endpoint
`/trainOnBatch`

### Parameters

#### Request Body

- **`x`** (Tensor | Tensor[] | Map<string, Tensor>) - Required - The input data for the batch.
- **`y`** (Tensor | Tensor[] | Map<string, Tensor>) - Required - The target data for the batch.

### Request Example
```json
{
  "x": [tensor1, tensor2],
  "y": [tensor3]
}
```

### Response

#### Success Response (200)
- **loss** (number) - The loss value for the batch.
- **metrics** (number[]) - An array of metric values for the batch.

#### Response Example
```json
{
  "loss": 0.5,
  "metrics": [0.85, 0.92]
}
```
```

--------------------------------

### Convert RGB to Grayscale

Source: https://js.tensorflow.org/api/latest/index

Converts images from RGB format to grayscale.

```APIDOC
## POST /tf.image.rgbToGrayscale

### Description
Converts images from RGB format to grayscale.

### Method
POST

### Endpoint
/tf.image.rgbToGrayscale

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **image** (tf.Tensor2D|tf.Tensor3D|tf.Tensor4D|tf.Tensor5D| tf.Tensor6D|TypedArray|Array) - Required - A RGB tensor to convert. The `image`'s last dimension must be size 3 with at least a two-dimensional shape.

### Request Example
```json
{
  "image": [[[ [1, 2, 3], [4, 5, 6] ]]]
}
```

### Response
#### Success Response (200)
- **grayscaleImage** (tf.Tensor2D|tf.Tensor3D|tf.Tensor4D|tf.Tensor5D| tf.Tensor6D) - The grayscale image.

#### Response Example
```json
{
  "grayscaleImage": [[[ [1.99, 1.99, 1.99], [4.99, 4.99, 4.99] ]]]
}
```
```

--------------------------------

### Train Model with tf.Model.fit

Source: https://js.tensorflow.org/api/latest/index

Trains a TensorFlow.js model for a fixed number of epochs (iterations on a dataset). This method requires input data, target data, and optional arguments like batch size, epochs, verbosity, and callbacks. It returns a Promise that resolves with training logs.

```javascript
const model = tf.sequential({
     layers: [tf.layers.dense({units: 1, inputShape: [10]})]
});
model.compile({optimizer: 'sgd', loss: 'meanSquaredError'});
for (let i = 1; i < 5 ; ++i) {
   const h = await model.fit(tf.ones([8, 10]), tf.ones([8, 1]), {
       batchSize: 4,
       epochs: 3
   });
   console.log("Loss after Epoch " + i + " : " + h.history.loss[0]);
}
```

--------------------------------

### tf.image.nonMaxSuppressionPaddedAsync

Source: https://js.tensorflow.org/api/latest/index

Asynchronously performs non-maximum suppression (NMS) on bounding boxes with padding, returning a promise.

```APIDOC
## tf.image.nonMaxSuppressionPaddedAsync

### Description
Asynchronously performs non maximum suppression of bounding boxes based on iou (intersection over union), with an option to pad results.

### Method
N/A (Function)

### Endpoint
N/A (Function)

### Parameters
#### Path Parameters
N/A

#### Query Parameters
N/A

#### Request Body
*   **boxes** (tf.Tensor2D|TypedArray|Array) - Required - A 2D tensor of shape `[numBoxes, 4]`. Each entry is `[y1, x1, y2, x2]`, where `(y1, x1)` and `(y2, x2)` are the corners of the bounding box.
*   **scores** (tf.Tensor1D|TypedArray|Array) - Required - A 1D tensor providing the box scores of shape `[numBoxes]`.
*   **maxOutputSize** (number) - Required - The maximum number of boxes to be selected.
*   **iouThreshold** (number) - Optional - A float representing the threshold for deciding whether boxes overlap too much with respect to IOU. Must be between [0, 1]. Defaults to 0.5 (50% box overlap).
*   **scoreThreshold** (number) - Optional - A threshold for deciding when to remove boxes based on score. Defaults to -inf, which means any score is accepted.
*   **padToMaxOutputSize** (boolean) - Optional - Defaults to false. If true, size of output `selectedIndices` is padded to maxOutputSize.

### Request Example
```json
{
  "boxes": "tf.Tensor2D",
  "scores": "tf.Tensor1D",
  "maxOutputSize": 10,
  "iouThreshold": 0.5,
  "scoreThreshold": 0.1,
  "padToMaxOutputSize": true
}
```

### Response
#### Success Response (200)
*   **Promise<{[name: string]: tf.Tensor}>** - A promise that resolves to an object containing tensors, potentially including `selectedIndices` padded to `maxOutputSize` if `padToMaxOutputSize` is true.

#### Response Example
```json
{
  "selectedIndices": "Promise<tf.Tensor1D (Potentially padded)>"
}
```
```

--------------------------------

### tf.initializers.varianceScaling

Source: https://js.tensorflow.org/api/latest/index

Initializer capable of adapting its scale to the shape of weights. With distribution=NORMAL, samples are drawn from a truncated normal distribution.

```APIDOC
## tf.initializers.varianceScaling (config)

### Description
Initializer capable of adapting its scale to the shape of weights. With distribution=NORMAL, samples are drawn from a truncated normal distribution centered on zero, with `stddev = sqrt(scale / n)` where n is:

- `fan_in` if `distribution` is 'fan_in' or 'uniform'.
- `fan_out` if `distribution` is 'fan_out'.
- `(fan_in + fan_out) / 2` if `distribution` is 'fan_avg'.

### Method
Function

### Endpoint
N/A (Client-side JavaScript function)

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
* **config** (Object) - Configuration object for the initializer.
* **scale** (number) - Optional. Scaling factor for the initializer. Defaults to 1.
* **mode** (string) - Optional. One of 'fan_in', 'fan_out', 'fan_avg'. Defaults to 'fan_in'.
* **distribution** (string) - Optional. One of 'uniform', 'normal'. Defaults to 'normal'.
* **seed** (number) - Optional. Random number generator seed.

### Request Example
```json
{
  "config": {
    "scale": 2.0,
    "mode": "fan_out",
    "distribution": "uniform",
    "seed": 107
  }
}
```

### Response
#### Success Response (tf.initializers.Initializer)
Returns an instance of tf.initializers.Initializer.

#### Response Example
```json
{
  "instance": "tf.initializers.Initializer"
}
```
```

--------------------------------

### Multi RNN Cell API

Source: https://js.tensorflow.org/api/latest/index

Computes the next states and outputs of a stack of LSTMCells.

```APIDOC
## tf.multiRNNCell

### Description
Computes the next states and outputs of a stack of LSTMCells. Each cell output is used as input to the next cell. Returns `[cellState, cellOutput]`. Derived from tf.contrib.rn.MultiRNNCell.

### Method
Not specified (assumed to be a function call)

### Endpoint
N/A

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **lstmCells** ((data: tf.Tensor2D, c: tf.Tensor2D, h: tf.Tensor2D): [tf.Tensor2D, tf.Tensor2D][]) - Required - Array of LSTMCell functions.
- **data** (tf.Tensor2D|TypedArray|Array) - Required - The input to the cell.
- **c** (Array) - Required - Array of previous cell states.
- **h** (Array) - Required - Array of previous cell outputs.

### Request Example
```javascript
const cells = [
  (data, c, h) => tf.basicLSTMCell(forgetBias, kernel1, bias1, data, c, h),
  (data, c, h) => tf.basicLSTMCell(forgetBias, kernel2, bias2, data, c, h)
];
const states = [tf.randomNormal([batchSize, hiddenSize1]), tf.randomNormal([batchSize, hiddenSize2])];
const outputs = [tf.randomNormal([batchSize, hiddenSize1]), tf.randomNormal([batchSize, hiddenSize2])];

tf.multiRNNCell(cells, inputData, states, outputs)
```

### Response
#### Success Response (200)
- **[cellState, cellOutput]** ([tf.Tensor2D[], tf.Tensor2D[]]) - Array containing the stacked cell states and stacked cell outputs.
```

--------------------------------

### Compute Log Sigmoid Element-wise with tf.logSigmoid

Source: https://js.tensorflow.org/api/latest/index

Computes the log sigmoid of the input tensor element-wise (`logSigmoid(x)`). For numerical stability, it uses `-tf.softplus(-x)`. It takes a tensor as input and returns a tensor with the computed values. Available for webgpu, webgl, wasm, and cpu backends.

```javascript
const x = tf.tensor1d([0, 1, -1, .7]);

x.logSigmoid().print();  // or tf.logSigmoid(x)
```

--------------------------------

### Create and Print Tensor with Ones

Source: https://js.tensorflow.org/api/latest/index

Creates a tensor with the same shape and dtype as the input tensor, filled with ones. The resulting tensor is then printed to the console. This function supports multiple backends including webgpu, webgl, wasm, and cpu.

```javascript
const x = tf.tensor([1, 2]);
tf.onesLike(x).print();
```

--------------------------------

### Space to Batch Conversion with tf.spaceToBatchND

Source: https://js.tensorflow.org/api/latest/index

Divides spatial dimensions of an input tensor into blocks and interleaves them with the batch dimension. Supports optional zero padding. Supports various backends like webgpu, webgl, wasm, and cpu.

```javascript
const x = tf.tensor4d([1, 2, 3, 4], [1, 2, 2, 1]);
const blockShape = [2, 2];
const paddings = [[0, 0], [0, 0]];

x.spaceToBatchND(blockShape, paddings).print();
```

--------------------------------

### Construct tf.RMSPropOptimizer in JavaScript

Source: https://js.tensorflow.org/api/latest/index

Constructs a tf.RMSPropOptimizer that uses RMSProp gradient descent. RMSProp divides the learning rate by an exponentially decaying average of squared gradients.

```javascript
const learningRate = 0.01;
const optimizer = tf.train.rmsprop(learningRate);

```

--------------------------------

### Enable Debug Mode in TensorFlow.js

Source: https://js.tensorflow.org/api/latest/index

Enables debug mode to log kernel execution details and tensor information. This mode significantly slows down applications and should not be used in production. It does not measure kernel download time.

```javascript
tf.enableDebugMode ();
```

--------------------------------

### tf.tile

Source: https://js.tensorflow.org/api/latest/index

Constructs a tensor by repeating it the number of times given by reps.

```APIDOC
## tf.tile

### Description
Constructs a tensor by repeating it the number of times given by `reps`. This operation creates a new tensor by replicating `input` `reps` times.

### Method
`tile(x: tf.Tensor | tf.TypedArray | tf.TensorLike, reps: number[]): tf.Tensor`

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
None

### Request Example
```javascript
const a = tf.tensor1d([1, 2]);
a.tile([2]).print(); // or tf.tile(a, [2])

const b = tf.tensor2d([1, 2, 3, 4], [2, 2]);
b.tile([1, 2]).print(); // or tf.tile(b, [1,2])
```

### Response
#### Success Response (200)
- **tf.Tensor** - The tiled tensor.

#### Response Example
```json
{
  "example": "tensor data"
}
```
```

--------------------------------

### Construct tf.SGDOptimizer in JavaScript

Source: https://js.tensorflow.org/api/latest/index

Constructs a tf.SGDOptimizer that uses stochastic gradient descent. This optimizer is suitable for training machine learning models by updating model weights based on the gradients calculated during backpropagation.

```javascript
const learningRate = 0.01;
const optimizer = tf.train.sgd(learningRate);

```

--------------------------------

### Create Tensor with Zeros of Same Shape

Source: https://js.tensorflow.org/api/latest/index

Creates a tensor with all elements set to zero and having the same shape as the provided input tensor. This is useful for initializing tensors based on existing ones. Compatible with webgpu, webgl, wasm, and cpu.

```javascript
const x = tf.tensor([1, 2]);
tf.zerosLike(x).print();
```

--------------------------------

### tf.variable

Source: https://js.tensorflow.org/api/latest/index

Creates a new variable with the provided initial value.

```APIDOC
## POST /tf.variable

### Description
Creates a new variable with the provided initial value.

### Method
POST

### Endpoint
/tf.variable

### Parameters
#### Query Parameters
- **initialValue** (tf.Tensor) - Required - Initial value for the tensor.
- **trainable** (boolean) - Optional - If true, optimizers are allowed to update it.
- **name** (string) - Optional - Name of the variable. Defaults to a unique id.
- **dtype** ('float32'|'int32'|'bool'|'complex64'|'string') - Optional - If set, initialValue will be converted to the given type.

### Request Example
```json
{
  "initialValue": [1, 2, 3],
  "trainable": true,
  "name": "myVariable"
}
```

### Response
#### Success Response (200)
- **tf.Variable** - The created variable.

#### Response Example
```json
{
  "variable": [1, 2, 3]
}
```
```

--------------------------------

### Create Tensor with Uniform Distribution Values (TensorFlow.js)

Source: https://js.tensorflow.org/api/latest/index

Generates a tensor with values sampled from a uniform distribution within a specified range [minval, maxval). Supports various backends like WebGPU, WebGL, WASM, and CPU. Input parameters include shape, minval, maxval, dtype, and an optional seed for reproducibility.

```javascript
tf.randomUniform([2, 2]).print();
```

--------------------------------

### tf.pad

Source: https://js.tensorflow.org/api/latest/index

Pads a tf.Tensor with a given value and paddings. This operation implements `CONSTANT` mode. For `REFLECT` and `SYMMETRIC`, refer to tf.mirrorPad().

```APIDOC
## tf.pad

### Description
Pads a tf.Tensor with a given value and paddings. This operation implements `CONSTANT` mode. For `REFLECT` and `SYMMETRIC`, refer to tf.mirrorPad().
Also available are stricter rank-specific methods with the same signature as this method that assert that `paddings` is of given length.
  * `tf.pad1d`
  * `tf.pad2d`
  * `tf.pad3d`
  * `tf.pad4d`

### Method
`tf.pad(x, paddings, constantValue?)`

### Parameters
#### Parameters
- **x** (tf.Tensor|TypedArray|Array) - The tensor to pad.
- **paddings** (Array) - An array of length `R` (the rank of the tensor), where each element is a length-2 tuple of ints `[padBefore, padAfter]`, specifying how much to pad along each dimension of the tensor. In "reflect" mode, the padded regions do not include the borders, while in "symmetric" mode the padded regions do include the borders. For example, if the input is `[1, 2, 3]` and paddings is `[0, 2]`, then the output is `[1, 2, 3, 2, 1]` in "reflect" mode, and `[1, 2, 3, 3, 2]` in "symmetric" mode. If `mode` is "reflect" then both `paddings[D, 0]` and `paddings[D, 1]` must be no greater than `x.shape[D] - 1`. If mode is "symmetric" then both `paddings[D, 0]` and `paddings[D, 1]` must be no greater than `x.shape[D]`.
- **constantValue** (number) - The pad value to use. Defaults to 0. Optional

### Request Example
```javascript
const x = tf.tensor1d([1, 2, 3, 4]);
x.pad([[1, 2]]).print();
```

### Response
#### Success Response (200)
- **output** (tf.Tensor) - The padded tensor.
```

--------------------------------

### tf.train.sgd

Source: https://js.tensorflow.org/api/latest/index

Constructs a tf.SGDOptimizer that uses stochastic gradient descent.

```APIDOC
## tf.train.sgd

### Description
Constructs a tf.SGDOptimizer that uses stochastic gradient descent.

### Method
`tf.train.sgd

### Endpoint
N/A (This is a function call within the library)

### Parameters
#### Path Parameters
N/A

#### Query Parameters
N/A

#### Request Body
N/A

### Request Example
```javascript
const learningRate = 0.01;
const optimizer = tf.train.sgd(learningRate);
```

### Response
#### Success Response (200)
Returns a `tf.SGDOptimizer` instance.

#### Response Example
```json
{
  "optimizerType": "sgd"
}
```
```

--------------------------------

### tf.train.rmsprop

Source: https://js.tensorflow.org/api/latest/index

Constructs a tf.RMSPropOptimizer that uses RMSProp gradient descent.

```APIDOC
## tf.train.rmsprop

### Description
Constructs a tf.RMSPropOptimizer that uses RMSProp gradient descent. This implementation uses plain momentum and is not centered version of RMSProp.

### Method
`tf.train.rmsprop`

### Endpoint
N/A (This is a function call within the library)

### Parameters
#### Path Parameters
N/A

#### Query Parameters
N/A

#### Request Body
N/A

### Request Example
```javascript
const learningRate = 0.01;
const optimizer = tf.train.rmsprop(learningRate);
```

### Response
#### Success Response (200)
Returns a `tf.RMSPropOptimizer` instance.

#### Response Example
```json
{
  "optimizerType": "rmsprop"
}
```
```

--------------------------------

### tf.train.adamax

Source: https://js.tensorflow.org/api/latest/index

Constructs a `tf.AdamaxOptimizer` that uses the Adamax algorithm.

```APIDOC
## tf.train.adamax

### Description
Constructs a `tf.AdamaxOptimizer` that uses the Adamax algorithm.

### Method
`tf.train.adamax`

### Endpoint
N/A (This is a function call within the library)

### Parameters
#### Path Parameters
N/A

#### Query Parameters
N/A

#### Request Body
N/A

### Request Example
```javascript
const learningRate = 0.01;
const optimizer = tf.train.adamax(learningRate);
```

### Response
#### Success Response (200)
Returns a `tf.AdamaxOptimizer` instance.

#### Response Example
```json
{
  "optimizerType": "adamax"
}
```
```

--------------------------------

### tf.io.copyModel

Source: https://js.tensorflow.org/api/latest/index

Copies a model from a source URL to a destination URL.

```APIDOC
## Copy Model (tf.io.copyModel)

### Description
Copies a model from one URL to another. This function supports copying within the same storage medium or between different storage mediums.

### Method
Asynchronous (uses Promises)

### Endpoint
N/A (operates on `sourceURL` and `destURL` arguments)

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
None

### Parameters
- **sourceURL** (string) - Required - The source URL of the model to copy.
- **destURL** (string) - Required - The destination URL where the model will be copied.

### Request Example
```javascript
// First, save a model to local storage
const model = tf.sequential();
model.add(tf.layers.dense({units: 1, inputShape: [10], activation: 'sigmoid'}));
await model.save('localstorage://demo/management/model1');

// Copy the model from local storage to IndexedDB
await tf.io.copyModel(
  'localstorage://demo/management/model1',
  'indexeddb://demo/management/model1'
);

console.log('Model copied successfully.');

// Clean up
await tf.io.removeModel('localstorage://demo/management/model1');
await tf.io.removeModel('indexeddb://demo/management/model1');
```

### Response
#### Success Response
- **Promise<ModelArtifactsInfo>** - A Promise that resolves with information about the copied model artifacts upon successful completion.
```

--------------------------------

### TensorFlow.js: Matrix Multiplication using einsum

Source: https://js.tensorflow.org/api/latest/index

Performs matrix multiplication using the `einsum` function, a powerful tool for tensor contractions based on Einstein summation. It takes two tensors and an equation string specifying the operation.

```javascript
const x = tf.tensor2d([[1, 2, 3], [4, 5, 6]]);
const y = tf.tensor2d([[0, 1], [2, 3], [4, 5]]);
x.print();
y.print();
tf.einsum('ij,jk->ik', x, y).print();
```

--------------------------------

### Manage Models with tf.io.listModels and tf.io.removeModel in JavaScript

Source: https://js.tensorflow.org/api/latest/index

Demonstrates saving a sequential model to local storage, listing all available models, and then removing a specific model. This function is useful for basic model management within a web browser environment using localStorage.

```javascript
const model = tf.sequential();
model.add(tf.layers.dense({units: 1, inputShape: [10], activation: 'sigmoid'}));
await model.save('localstorage://demo/management/model1');

console.log(JSON.stringify(await tf.io.listModels()));

await tf.io.removeModel('localstorage://demo/management/model1');

console.log(JSON.stringify(await tf.io.listModels()));
```

--------------------------------

### shuffle (bufferSize, seed?, reshuffleEachIteration?)

Source: https://js.tensorflow.org/api/latest/index

Pseudorandomly shuffles the elements of this dataset. This is done in a streaming manner, by sampling from a given number of prefetched elements.

```APIDOC
## shuffle (bufferSize, seed?, reshuffleEachIteration?)

### Description
Pseudorandomly shuffles the elements of this dataset. This is done in a streaming manner, by sampling from a given number of prefetched elements.

### Method
shuffle

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **bufferSize** (number) - An integer specifying the number of elements from this dataset from which the new dataset will sample.
- **seed** (string) - (Optional) An integer specifying the random seed that will be used to create the distribution.
- **reshuffleEachIteration** (boolean) - (Optional) A boolean, which if true indicates that the dataset should be pseudorandomly reshuffled each time it is iterated over. If false, elements will be returned in the same shuffled order on each iteration. (Defaults to `true`.)

### Request Example
```javascript
const a = tf.data.array([1, 2, 3, 4, 5, 6]).shuffle(3);
a.forEachAsync(e => console.log(e));
```

### Response
#### Success Response (200)
- **tf.data.Dataset** - The shuffled dataset.

#### Response Example
```json
{
  "example": "Dataset object"
}
```
```

--------------------------------

### Basic LSTM Cell API

Source: https://js.tensorflow.org/api/latest/index

Computes the next state and output of a BasicLSTMCell.

```APIDOC
## tf.basicLSTMCell

### Description
Computes the next state and output of a BasicLSTMCell. Returns `[newC, newH]`. Derived from tf.contrib.rnn.BasicLSTMCell.

### Method
Not specified (assumed to be a function call)

### Endpoint
N/A

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **forgetBias** (tf.Scalar|TypedArray|Array) - Required - Forget bias for the cell.
- **lstmKernel** (tf.Tensor2D|TypedArray|Array) - Required - The weights for the cell.
- **lstmBias** (tf.Tensor1D|TypedArray|Array) - Required - The bias for the cell.
- **data** (tf.Tensor2D|TypedArray|Array) - Required - The input to the cell.
- **c** (tf.Tensor2D|TypedArray|Array) - Required - Previous cell state.
- **h** (tf.Tensor2D|TypedArray|Array) - Required - Previous cell output.

### Request Example
```javascript
const forgetBias = tf.scalar(0.5);
const lstmKernel = tf.randomNormal([inputSize + hiddenSize, 4 * hiddenSize]);
const lstmBias = tf.randomNormal([4 * hiddenSize]);
const data = tf.randomNormal([batchSize, inputSize]);
const c = tf.randomNormal([batchSize, hiddenSize]);
const h = tf.randomNormal([batchSize, hiddenSize]);

tf.basicLSTMCell(forgetBias, lstmKernel, lstmBias, data, c, h)
```

### Response
#### Success Response (200)
- **[newC, newH]** ([tf.Tensor2D, tf.Tensor2D]) - Array containing the new cell state and new cell output.
```

--------------------------------

### Shuffle Dataset Elements Randomly with JavaScript

Source: https://js.tensorflow.org/api/latest/index

The `shuffle` method creates a dataset that pseudorandomly shuffles the elements of the original dataset. Shuffling is performed in a streaming manner by sampling from a buffer of prefetched elements. Optionally, a seed can be provided for reproducible shuffling, and `reshuffleEachIteration` controls whether to reshuffle on each iteration.

```javascript
const a = tf.data.array([1, 2, 3, 4, 5, 6]).shuffle(3);
await a.forEachAsync(e => console.log(e));

```

--------------------------------

### Apply 2D Zero Padding in TensorFlow.js

Source: https://js.tensorflow.org/api/latest/index

Illustrates the usage of the `tf.layers.zeroPadding2d` layer to add rows and columns of zeros to a 2D input tensor, commonly used for image processing. It explains how the `padding` argument can be configured to apply symmetric or asymmetric padding to height and width.

```javascript
// Example of using zeroPadding2d (specific code not provided in source, conceptual usage):
// const paddedLayer = tf.layers.zeroPadding2d({padding: [[2, 2], [2, 2]]});
// const inputTensor = tf.randomNormal([1, 10, 10, 3]); // Example input
// const outputTensor = paddedLayer.apply(inputTensor);
// outputTensor.print();
```

--------------------------------

### Compute Hyperbolic Sine of Tensor Elements (JavaScript)

Source: https://js.tensorflow.org/api/latest/index

Computes the element-wise hyperbolic sine of a tensor. Supports CPU, WebGL, WebGPU, and WASM backends. Accepts a tensor, TypedArray, or Array as input and returns a tensor.

```javascript
const x = tf.tensor1d([0, 1, -1, .7]);

x.sinh().print();  // or tf.sinh(x)
```

--------------------------------

### Construct tf.AdamOptimizer in JavaScript

Source: https://js.tensorflow.org/api/latest/index

Constructs a tf.AdamOptimizer that uses the Adam algorithm. Adam is an optimization algorithm that computes adaptive learning rates for each parameter.

```javascript
const learningRate = 0.001;
const optimizer = tf.train.adam(learningRate);

```

--------------------------------

### Create Sequential Model (TensorFlow.js)

Source: https://js.tensorflow.org/api/latest/index

Constructs a sequential model in TensorFlow.js, suitable for linear stacks of layers. The first layer must have an input shape defined. Supports WebGPU, WebGL, WASM, and CPU backends. Models can be created by adding layers individually or by providing an array of layers in the configuration.

```javascript
const model = tf.sequential();

// First layer must have an input shape defined.
model.add(tf.layers.dense({units: 32, inputShape: [50]}));
// Afterwards, TF.js does automatic shape inference.
model.add(tf.layers.dense({units: 4}));

// Inspect the inferred shape of the model's output, which equals
// `[null, 4]`. The 1st dimension is the undetermined batch dimension;
// the 2nd is the output size of the model's last layer.
console.log(JSON.stringify(model.outputs[0].shape));
```

```javascript
const model = tf.sequential();

// First layer must have a defined input shape
model.add(tf.layers.dense({units: 32, batchInputShape: [null, 50]}));
// Afterwards, TF.js does automatic shape inference.
model.add(tf.layers.dense({units: 4}));

// Inspect the inferred shape of the model's output.
console.log(JSON.stringify(model.outputs[0].shape));
```

```javascript
const model = tf.sequential({
   layers: [tf.layers.dense({units: 32, inputShape: [50]}),
            tf.layers.dense({units: 4})]
});
console.log(JSON.stringify(model.outputs[0].shape));
```

--------------------------------

### tf.io.listModels

Source: https://js.tensorflow.org/api/latest/index

Lists all models stored in the registered storage mediums.

```APIDOC
## List Models (tf.io.listModels)

### Description
Lists all models currently stored across all registered storage mediums (e.g., Local Storage, IndexedDB).

### Method
Asynchronous (uses Promises)

### Endpoint
N/A

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
None

### Request Example
```javascript
// Assuming models have been saved previously
const modelInfo = await tf.io.listModels();
console.log(JSON.stringify(modelInfo));
```

### Response
#### Success Response
- **Promise<Object>** - A Promise that resolves with an object containing information about all stored models. The keys are model URLs and the values are objects with `modelArtifactsInfo` and `dateSaved`.

#### Response Example
```json
{
  "localstorage://demo/management/model1": {
    "modelArtifactsInfo": {
      "dateSaved": "2023-10-27T10:00:00.000Z",
      "modelTopologyType": "nothing",
      "weightSpecrzostype": "nothing"
    },
    "dateSaved": "2023-10-27T10:00:00.000Z"
  }
}
```
```

--------------------------------

### tf.ones

Source: https://js.tensorflow.org/api/latest/index

Creates a tensor with all elements set to 1.

```APIDOC
## POST /websites/js_tensorflow_api/tf.ones

### Description
Creates a tf.Tensor with all elements set to 1.

### Method
POST

### Endpoint
/websites/js_tensorflow_api/tf.ones

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **shape** (number[]) - An array of integers defining the output tensor shape.
- **dtype** ('float32'|'int32'|'bool'|'complex64'|'string') - The type of an element in the resulting tensor. Defaults to 'float'. Optional

### Request Example
```json
{
  "shape": [2, 2]
}
```

### Response
#### Success Response (200)
- **ones_tensor** (tf.Tensor) - A tensor with all elements set to 1.

#### Response Example
```json
{
  "ones_tensor": [[1, 1], [1, 1]]
}
```
```

--------------------------------

### Batch Normalization Layer

Source: https://js.tensorflow.org/api/latest/index

Implements Batch Normalization, a technique to normalize layer activations.

```APIDOC
## POST /tf.layers.batchNormalization

### Description
Batch normalization layer (Ioffe and Szegedy, 2014). Normalize the activations of the previous layer at each batch, i.e. applies a transformation that maintains the mean activation close to 0 and the activation standard deviation close to 1.

### Method
POST

### Endpoint
/tf.layers.batchNormalization

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **args** (Object) - Optional
- **axis** (number) - Optional - The integer axis that should be normalized (typically the features axis). Defaults to -1.
- **momentum** (number) - Optional - Momentum of the moving average. Defaults to 0.99.
- **epsilon** (number) - Optional - Small float added to the variance to avoid dividing by zero. Defaults to 1e-3.
- **center** (boolean) - Optional - If `true`, add offset of `beta` to normalized tensor. If `false`, `beta` is ignored. Defaults to `true`.
- **scale** (boolean) - Optional - If `true`, multiply by `gamma`. If `false`, `gamma` is not used. When the next layer is linear (also e.g. `nn.relu`), this can be disabled since the scaling will be done by the next layer. Defaults to `true`.
- **betaInitializer** (string|tf.initializers.Initializer) - Optional - Initializer for the beta weight. Defaults to 'zeros'.
- **gammaInitializer** (string|tf.initializers.Initializer) - Optional - Initializer for the gamma weight. Defaults to `ones`.
- **movingMeanInitializer** (string|tf.initializers.Initializer) - Optional - Initializer for the moving mean. Defaults to `zeros`.
- **movingVarianceInitializer** (string|tf.initializers.Initializer) - Optional - Initializer for the moving variance. Defaults to 'Ones'.
- **betaConstraint** (string|tf.constraints.Constraint) - Optional - Constraint for the beta weight.
- **gammaConstraint** (string|tf.constraints.Constraint) - Optional - Constraint for gamma weight.
- **betaRegularizer** (string|Regularizer) - Optional - Regularizer for the beta weight.
- **gammaRegularizer** (string|Regularizer) - Optional - Regularizer for the gamma weight.
- **inputShape** (Array<number|null>) - Optional - If defined, will be used to create an input layer to insert before this layer. If both `inputShape` and `batchInputShape` are defined, `batchInputShape` will be used. This argument is only applicable to input layers (the first layer of a model).

### Request Example
```json
{
  "axis": -1,
  "momentum": 0.99,
  "epsilon": 0.001,
  "center": true,
  "scale": true,
  "betaInitializer": "zeros",
  "gammaInitializer": "ones"
}
```

### Response
#### Success Response (200)
- **layer** (object) - The configured BatchNormalization layer.

#### Response Example
```json
{
  "layer": {
    "name": "batchNormalization",
    "axis": -1,
    "momentum": 0.99,
    "epsilon": 0.001,
    "center": true,
    "scale": true,
    "betaInitializer": {"config": {"value": 0}},
    "gammaInitializer": {"config": {"value": 1}},
    "movingMeanInitializer": {"config": {"value": 0}},
    "movingVarianceInitializer": {"config": {"value": 1}},
    "betaConstraint": null,
    "gammaConstraint": null,
    "betaRegularizer": null,
    "gammaRegularizer": null
  }
}
```
```

--------------------------------

### Compute Natural Logarithm Element-wise with tf.log

Source: https://js.tensorflow.org/api/latest/index

Computes the natural logarithm of the input tensor element-wise (`ln(x)`). It takes a tensor as input and returns a tensor with the computed values. Available for webgpu, webgl, wasm, and cpu backends.

```javascript
const x = tf.tensor1d([1, 2, Math.E]);

x.log().print();  // or tf.log(x)
```

--------------------------------

### tf.image.nonMaxSuppressionAsync

Source: https://js.tensorflow.org/api/latest/index

Asynchronous version of `tf.image.nonMaxSuppression`. Performs non-maximum suppression (NMS) on bounding boxes.

```APIDOC
## tf.image.nonMaxSuppressionAsync

### Description
Performs non maximum suppression of bounding boxes based on iou (intersection over union). This is the async version of `nonMaxSuppression`.

### Method
N/A (Function)

### Endpoint
N/A (Function)

### Parameters
#### Path Parameters
N/A

#### Query Parameters
N/A

#### Request Body
*   **boxes** (tf.Tensor2D|TypedArray|Array) - Required - A 2D tensor of shape `[numBoxes, 4]`. Each entry is `[y1, x1, y2, x2]`, where `(y1, x1)` and `(y2, x2)` are the corners of the bounding box.
*   **scores** (tf.Tensor1D|TypedArray|Array) - Required - A 1D tensor providing the box scores of shape `[numBoxes]`.
*   **maxOutputSize** (number) - Required - The maximum number of boxes to be selected.
*   **iouThreshold** (number) - Optional - A float representing the threshold for deciding whether boxes overlap too much with respect to IOU. Must be between [0, 1]. Defaults to 0.5 (50% box overlap).
*   **scoreThreshold** (number) - Optional - A threshold for deciding when to remove boxes based on score. Defaults to -inf, which means any score is accepted.

### Request Example
```json
{
  "boxes": "tf.Tensor2D",
  "scores": "tf.Tensor1D",
  "maxOutputSize": 10,
  "iouThreshold": 0.5,
  "scoreThreshold": 0.1
}
```

### Response
#### Success Response (200)
*   **Promise<tf.Tensor1D>** - A promise that resolves to a 1D tensor containing the indices of the selected boxes.

#### Response Example
```json
{
  "selectedIndices": "Promise<tf.Tensor1D>"
}
```
```

--------------------------------

### tf.train.adadelta

Source: https://js.tensorflow.org/api/latest/index

Constructs a tf.AdadeltaOptimizer that uses the Adadelta algorithm.

```APIDOC
## tf.train.adadelta

### Description
Constructs a tf.AdadeltaOptimizer that uses the Adadelta algorithm.

### Method
`tf.train.adadelta`

### Endpoint
N/A (This is a function call within the library)

### Parameters
#### Path Parameters
N/A

#### Query Parameters
N/A

#### Request Body
N/A

### Request Example
```javascript
const learningRate = 0.01;
const optimizer = tf.train.adadelta(learningRate);
```

### Response
#### Success Response (200)
Returns a `tf.AdadeltaOptimizer` instance.

#### Response Example
```json
{
  "optimizerType": "adadelta"
}
```
```

--------------------------------

### tf.minimum

Source: https://js.tensorflow.org/api/latest/index

Returns the min of a and b (`a < b ? a : b`) element-wise. Supports broadcasting.

```APIDOC
## tf.minimum

### Description
Returns the min of a and b (`a < b ? a : b`) element-wise. Supports broadcasting.

### Method
`tf.minimum(a, b)` or `a.minimum(b)`

### Parameters
- **a** (tf.Tensor|TypedArray|Array) - The first tensor.
- **b** (tf.Tensor|TypedArray|Array) - The second tensor. Must have the same type as `a`.

### Request Example
```javascript
const a = tf.tensor1d([1, 4, 3, 16]);
const b = tf.tensor1d([1, 2, 9, 4]);

a.minimum(b).print();  // or tf.minimum(a, b)
```

### Response
#### Success Response (tf.Tensor)
- **result** (tf.Tensor) - The element-wise minimum of a and b.

#### Response Example
```json
// Example output for a.minimum(b):
// Tensor
//    [1, 2, 3, 4]
```
```

--------------------------------

### Configuring RNN Layer Output and State Return in TensorFlow.js

Source: https://js.tensorflow.org/api/latest/index

This snippet illustrates how to control the output of an RNN layer in TensorFlow.js using `returnSequences` and `returnState` options. `returnSequences: true` provides the output for each time step, while `returnState: true` includes the final hidden state and cell state (if applicable) in the output.

```javascript
// Returns only the last output
const rnnLayer1 = tf.layers.lstm({
  units: 10,
  returnSequences: false,
  returnState: false
});

// Returns the full sequence of outputs
const rnnLayer2 = tf.layers.lstm({
  units: 10,
  returnSequences: true,
  returnState: false
});

// Returns the last output and the states
const rnnLayer3 = tf.layers.lstm({
  units: 10,
  returnSequences: false,
  returnState: true
});

// Returns the full sequence and the states
const rnnLayer4 = tf.layers.lstm({
  units: 10,
  returnSequences: true,
  returnState: true
});
```

--------------------------------

### tf.image.nonMaxSuppressionPadded

Source: https://js.tensorflow.org/api/latest/index

Asynchronously performs non-maximum suppression (NMS) on bounding boxes with an option to pad the results to a specified size.

```APIDOC
## tf.image.nonMaxSuppressionPadded

### Description
Asynchronously performs non maximum suppression of bounding boxes based on iou (intersection over union), with an option to pad results.

### Method
N/A (Function)

### Endpoint
N/A (Function)

### Parameters
#### Path Parameters
N/A

#### Query Parameters
N/A

#### Request Body
*   **boxes** (tf.Tensor2D|TypedArray|Array) - Required - A 2D tensor of shape `[numBoxes, 4]`. Each entry is `[y1, x1, y2, x2]`, where `(y1, x1)` and `(y2, x2)` are the corners of the bounding box.
*   **scores** (tf.Tensor1D|TypedArray|Array) - Required - A 1D tensor providing the box scores of shape `[numBoxes]`.
*   **maxOutputSize** (number) - Required - The maximum number of boxes to be selected.
*   **iouThreshold** (number) - Optional - A float representing the threshold for deciding whether boxes overlap too much with respect to IOU. Must be between [0, 1]. Defaults to 0.5 (50% box overlap).
*   **scoreThreshold** (number) - Optional - A threshold for deciding when to remove boxes based on score. Defaults to -inf, which means any score is accepted.
*   **padToMaxOutputSize** (boolean) - Optional - Defaults to false. If true, size of output `selectedIndices` is padded to maxOutputSize.

### Request Example
```json
{
  "boxes": "tf.Tensor2D",
  "scores": "tf.Tensor1D",
  "maxOutputSize": 10,
  "iouThreshold": 0.5,
  "scoreThreshold": 0.1,
  "padToMaxOutputSize": true
}
```

### Response
#### Success Response (200)
*   **{[name: string]: tf.Tensor}** - An object containing tensors, potentially including `selectedIndices` padded to `maxOutputSize` if `padToMaxOutputSize` is true.

#### Response Example
```json
{
  "selectedIndices": "tf.Tensor1D (Potentially padded)"
}
```
```

--------------------------------

### Create Rank-5 Tensor with Nested Array - tf.tensor5d

Source: https://js.tensorflow.org/api/latest/index

Initializes a rank-5 tensor using a nested array. tf.tensor5d() is recommended for better code readability than tf.tensor().

```javascript
// Pass a nested array.
tf.tensor5d([[[[[1],[2]],[[3],[4]]],[[[5],[6]],[[7],[8]]]]]).print();

```

--------------------------------

### Compute 1D Real Fast Fourier Transform - tf.spectral.rfft

Source: https://js.tensorflow.org/api/latest/index

Computes the 1-dimensional discrete Fourier transform over the inner-most dimension of a real input tensor. An optional `fftLength` can be provided. Supported on WebGPU, WebGL, WASM, and CPU.

```javascript
const real = tf.tensor1d([1, 2, 3]);

real.rfft().print();

```

--------------------------------

### tf.initializers.heUniform

Source: https://js.tensorflow.org/api/latest/index

The He uniform initializer. It draws samples from a uniform distribution within a calculated limit based on the number of input units.

```APIDOC
## tf.initializers.heUniform (args)

### Description
He uniform initializer. It draws samples from a uniform distribution within [-limit, limit] where `limit` is `sqrt(6 / fan_in)` where `fanIn` is the number of input units in the weight tensor.

### Method
Function

### Endpoint
N/A (Client-side JavaScript function)

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
* **args** (Object) - Configuration object for the initializer.
* **seed** (number) - Optional. Random number generator seed.

### Request Example
```json
{
  "args": {},
  "seed": 789
}
```

### Response
#### Success Response (tf.initializers.Initializer)
Returns an instance of tf.initializers.Initializer.

#### Response Example
```json
{
  "instance": "tf.initializers.Initializer"
}
```

### Reference
He et al., http://arxiv.org/abs/1502.01852
```

--------------------------------

### Create N-grams from Ragged String Data with TensorFlow.js

Source: https://js.tensorflow.org/api/latest/index

Generates n-grams from ragged string data. This function takes string data, split information, a separator, n-gram widths, and padding options. It outputs a ragged tensor containing the n-grams. Available for WebGPU, WebGL, WASM, and CPU.

```javascript
const result = tf.string.stringNGrams(
   ['a', 'b', 'c', 'd'], tf.tensor1d([0, 2, 4], 'int32'),
   '|', [1, 2], 'LP', 'RP', -1, false);
result['nGrams'].print(); // ['a', 'b', 'LP|a', 'a|b', 'b|RP',
                           //  'c', 'd', 'LP|c', 'c|d', 'd|RP']
result['nGramsSplits'].print(); // [0, 5, 10]
```

--------------------------------

### tf.min

Source: https://js.tensorflow.org/api/latest/index

Computes the minimum value from the input tensor.

```APIDOC
## tf.min

### Description
Computes the minimum value from the input tensor. Reduces the input along the dimensions given in `axis`. Unless `keepDims` is true, the rank of the tf.Tensor is reduced by 1 for each entry in `axis`. If `keepDims` is true, the reduced dimensions are retained with length 1. If `axis` has no entries, all dimensions are reduced, and a tf.Tensor with a single element is returned.

### Method
`tf.min(x, axis?, keepDims?)
`

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **x** (tf.Tensor|TypedArray|Array) - Required - The input tensor.
- **axis** (number|number[]) - Optional - The dimension(s) to reduce. By default it reduces all dimensions.
- **keepDims** (boolean) - Optional - If true, retains reduced dimensions with size 1.

### Request Example
```javascript
const x = tf.tensor1d([1, 2, 3]);
x.min().print();  // or tf.min(x)
```

### Response
#### Success Response (200)
- **result** (tf.Tensor) - The tensor with minimum values.

#### Response Example
```json
{
  "example": "tf.Tensor"
}
```
```

--------------------------------

### TensorFlow.js: Dot Product using einsum

Source: https://js.tensorflow.org/api/latest/index

Calculates the dot product of two 1D tensors using `einsum`. The equation 'i,i->' indicates that the elements at the same index 'i' are multiplied and then summed to a single scalar.

```javascript
const x = tf.tensor1d([1, 2, 3]);
const y = tf.tensor1d([0, 1, 2]);
x.print();
y.print();
tf.einsum('i,i->', x, y).print();
```

--------------------------------

### tf.initializers.heNormal

Source: https://js.tensorflow.org/api/latest/index

The He normal initializer. It draws samples from a truncated normal distribution centered on 0 with a standard deviation determined by the number of input units.

```APIDOC
## tf.initializers.heNormal (args)

### Description
He normal initializer. It draws samples from a truncated normal distribution centered on 0 with `stddev = sqrt(2 / fanIn)` where `fanIn` is the number of input units in the weight tensor.

### Method
Function

### Endpoint
N/A (Client-side JavaScript function)

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
* **args** (Object) - Configuration object for the initializer.
* **seed** (number) - Optional. Random number generator seed.

### Request Example
```json
{
  "args": {},
  "seed": 456
}
```

### Response
#### Success Response (tf.initializers.Initializer)
Returns an instance of tf.initializers.Initializer.

#### Response Example
```json
{
  "instance": "tf.initializers.Initializer"
}
```

### Reference
He et al., http://arxiv.org/abs/1502.01852
```

--------------------------------

### Compute Exponential Minus One Element-wise with tf.expm1

Source: https://js.tensorflow.org/api/latest/index

Computes the exponential of the input tensor minus one element-wise (`e ^ x - 1`). It takes a tensor as input and returns a tensor with the computed values. Available for webgpu, webgl, wasm, and cpu backends.

```javascript
const x = tf.tensor1d([1, 2, -3]);

x.expm1().print();  // or tf.expm1(x)
```

--------------------------------

### tf.add

Source: https://js.tensorflow.org/api/latest/index

Adds two tf.Tensors element-wise, A + B. Supports broadcasting.

```APIDOC
## tf.add

### Description
Adds two tf.Tensors element-wise, A + B. Supports broadcasting.

### Method
`tf.add(a, b)`

### Parameters
* **a** (tf.Tensor | TypedArray | Array) - The first tf.Tensor to add.
* **b** (tf.Tensor | TypedArray | Array) - The second tf.Tensor to add. Must have the same type as `a`.

### Request Example
```javascript
const a = tf.tensor1d([1, 2, 3, 4]);
const b = tf.tensor1d([10, 20, 30, 40]);

a.add(b).print();  // or tf.add(a, b)
```

### Response
#### Success Response (200)
* **tf.Tensor**: The resulting tensor after addition.

#### Response Example
```json
{
  "data": [11, 22, 33, 44],
  "shape": [4],
  "dtype": "float32"
}
```
```

--------------------------------

### Create Tensor with Uniform Integers (TensorFlow.js)

Source: https://js.tensorflow.org/api/latest/index

Creates a tensor containing integers sampled from a uniform distribution in the range [minval, maxval). Compatible with multiple execution backends (WebGPU, WebGL, WASM, CPU). Parameters include shape, minval, maxval, and an optional seed.

```javascript
tf.randomUniformInt([2, 2], 0, 10).print();
```

--------------------------------

### tf.addN

Source: https://js.tensorflow.org/api/latest/index

Adds a list of tf.Tensors element-wise, each with the same shape and dtype.

```APIDOC
## tf.addN

### Description
Adds a list of tf.Tensors element-wise, each with the same shape and dtype.

### Method
`tf.addN(tensors)`

### Parameters
* **tensors** (Array<tf.Tensor | TypedArray | Array>) - A list of tensors to add. All tensors must have the same shape and dtype.

### Request Example
```javascript
const a = tf.tensor1d([1, 2]);
const b = tf.tensor1d([3, 4]);
const c = tf.tensor1d([5, 6]);

tf.addN([a, b, c]).print();
```

### Response
#### Success Response (200)
* **tf.Tensor**: The resulting tensor after adding all tensors in the list.

#### Response Example
```json
{
  "data": [9, 12],
  "shape": [2],
  "dtype": "float32"
}
```
```

--------------------------------

### Compute Reciprocal Element-wise with tf.reciprocal

Source: https://js.tensorflow.org/api/latest/index

Computes the reciprocal of the input tensor element-wise (`1 / x`). It takes a tensor as input and returns a tensor with the computed values. Available for webgpu, webgl, wasm, and cpu backends.

```javascript
const x = tf.tensor1d([1, 2, 4]);

x.reciprocal().print(); // or tf.reciprocal(x)
```

--------------------------------

### Frame Input Tensor using TensorFlow.js

Source: https://js.tensorflow.org/api/latest/index

Expands an input tensor into frames of a specified length, sliding a window with a given step. This function is useful for signal processing tasks. It supports padding at the end of the signal if necessary and can be executed on multiple backends.

```javascript
tf.signal.frame([1, 2, 3], 2, 1).print();
```

--------------------------------

### Print Tensor with Verbose Option

Source: https://js.tensorflow.org/api/latest/index

Prints information about a tf.Tensor, including its data. The 'verbose' option, when set to true, provides additional details such as dtype and size. This function is available across multiple backends.

```javascript
const verbose = true;
tf.tensor2d([1, 2, 3, 4], [2, 2]).print(verbose);
```

--------------------------------

### Configure Input Layer with BatchNormalization in TensorFlow.js

Source: https://js.tensorflow.org/api/latest/index

This snippet demonstrates how to configure an input layer for batch normalization within a TensorFlow.js model. It highlights parameters like inputShape, batchSize, and dtype, which are crucial for defining the input's characteristics before normalization.

```javascript
const model = tf.sequential();
model.add(tf.layers.batchNormalization({
  inputShape: [784],
  batchSize: 32,
  dtype: 'float32',
  axis: -1,
  momentum: 0.99,
  epsilon: 1e-3,
  center: true,
  scale: true,
  betaInitializer: 'zeros',
  gammaInitializer: 'ones',
  movingMeanInitializer: 'zeros',
  movingVarianceInitializer: 'Ones',
  betaConstraint: null,
  gammaConstraint: null,
  betaRegularizer: null,
  gammaRegularizer: null
}));

// Example of using batchInputShape instead of inputShape and batchSize
model.add(tf.layers.batchNormalization({
  batchInputShape: [null, 784],
  dtype: 'float32'
}));
```

--------------------------------

### Rescaling Layer

Source: https://js.tensorflow.org/api/latest/index

A preprocessing layer which rescales input values to a new range.

```APIDOC
## tf.layers.rescaling

### Description
A preprocessing layer which rescales input values to a new range. This layer rescales every value of an input (often an image) by multiplying by `scale` and adding `offset`. For instance: 1. To rescale an input in the `[0, 255]` range to be in the `[0, 1]` range, you would pass `scale=1/255`. 2. To rescale an input in the `[0, 255]` range to be in the `[-1, 1]` range, you would pass `scale=1./127.5, offset=-1`. The rescaling is applied both during training and inference. Inputs can be of integer or floating point dtype, and by default the layer will output floats.

### Method
Not applicable (Layer constructor)

### Endpoint
Not applicable (Layer constructor)

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
None

##### Parameters for Layer Construction
- **args** (Object) - Optional
- **scale** (number) - The scale to apply to the inputs.
- **offset** (number) - The offset to apply to the inputs.
- **inputShape** (Array<number | null>) - If defined, will be used to create an input layer to insert before this layer. If both `inputShape` and `batchInputShape` are defined, `batchInputShape` will be used. This argument is only applicable to input layers (the first layer of a model).
- **batchInputShape** (Array<number | null>) - If defined, will be used to create an input layer to insert before this layer. If both `inputShape` and `batchInputShape` are defined, `batchInputShape` will be used. This argument is only applicable to input layers (the first layer of a model).
- **batchSize** (number) - If `inputShape` is specified and `batchInputShape` is _not_ specified, `batchSize` is used to construct the `batchInputShape`: `[batchSize, ...inputShape]`
- **dtype** ('float32'|'int32'|'bool'|'complex64'|'string') - The data-type for this layer. Defaults to 'float32'. This argument is only applicable to input layers (the first layer of a model).
- **name** (string) - Name for this layer.
- **trainable** (boolean) - Whether the weights of this layer are updatable by `fit`. Defaults to true.
- **weights** (Array<tf.Tensor>) - Initial weight values of the layer.

### Request Example
```json
{
  "example": "// TensorFlow.js code for creating Rescaling layer to normalize to [0, 1]\nconst layer = tf.layers.rescaling({scale: 1/255});"
}
```

### Response
#### Success Response (Layer Object)
- **Layer Object** (object) - A TensorFlow.js layer object configured for rescaling.

#### Response Example
```json
{
  "example": "// Rescaling layer object"
}
```
```

--------------------------------

### Construct tf.AdadeltaOptimizer in JavaScript

Source: https://js.tensorflow.org/api/latest/index

Constructs a tf.AdadeltaOptimizer that uses the Adadelta algorithm. Adadelta is an extension of Adagrad that attempts to reduce Adagrad's aggressive monotonically decreasing learning rate.

```javascript
const learningRate = 0.01;
const optimizer = tf.train.adadelta(learningRate);

```

--------------------------------

### Compute Softplus of Tensor Elements (JavaScript)

Source: https://js.tensorflow.org/api/latest/index

Computes the element-wise softplus function of a tensor, defined as log(exp(x) + 1). Supports CPU, WebGL, WebGPU, and WASM backends. Takes a tensor, TypedArray, or Array as input and returns a tensor.

```javascript
const x = tf.tensor1d([0, 1, -1, .7]);

x.softplus().print();  // or tf.softplus(x)
```

--------------------------------

### Compute element-wise ceiling using TensorFlow.js

Source: https://js.tensorflow.org/api/latest/index

Computes the ceiling of the input tensor element-wise. The `ceil` function takes a tensor `x` and returns a new tensor with the ceiling values (smallest integer greater than or equal to each element).

```javascript
const x = tf.tensor1d([.6, 1.1, -3.3]);

x.ceil().print();  // or tf.ceil(x)
```

--------------------------------

### Compute Gradients w.r.t. Trainable Variables with tf.variableGrads

Source: https://js.tensorflow.org/api/latest/index

Computes the gradient of a function `f` with respect to a specified list of trainable variables (`varList`). If `varList` is not provided, it defaults to all trainable variables in the current scope. The function `f` must return a scalar value. This function is available for webgpu, webgl, wasm, and cpu.

```javascript
const a = tf.variable(tf.tensor1d([3, 4]));
const b = tf.variable(tf.tensor1d([5, 6]));
const x = tf.tensor1d([1, 2]);

const f = () => a.mul(x.square()).add(b.mul(x)).sum();
const {value, grads} = tf.variableGrads(f);

Object.keys(grads).forEach(varName => grads[varName].print());

```

--------------------------------

### tf.upperBound

Source: https://js.tensorflow.org/api/latest/index

Searches for where a value would go in a sorted sequence, returning the bucket-index for each value.

```APIDOC
## tf.upperBound

### Description
Searches for where a value would go in a sorted sequence. This operation is typically used for "binning", "bucketing", or "discretizing". The index returned corresponds to the first edge greater than the value. It always operates on the innermost dimension (axis=-1).

### Method
`tf.upperBound(sortedSequence, values)
`

### Parameters
- **sortedSequence** (tf.Tensor | TypedArray | Array) - N-D. The sorted sequence along the innermost axis.
- **values** (tf.Tensor | TypedArray | Array) - N-D. The search values.

### Returns
- **tf.Tensor** - A tensor containing the bucket-indices for each value.

### Request Example
```javascript
const seq = tf.tensor1d([0, 3, 9, 10, 10]);
const values = tf.tensor1d([0, 4, 10]);
const result = tf.upperBound(seq, values);
result.print(); // [1, 2, 5]
```
```

--------------------------------

### tf.div

Source: https://js.tensorflow.org/api/latest/index

Divides two tf.Tensors element-wise, A / B. Supports broadcasting.

```APIDOC
## tf.div

### Description
Divides two tf.Tensors element-wise, A / B. Supports broadcasting.

### Method
`tf.div(a, b)`

### Parameters
* **a** (tf.Tensor | TypedArray | Array) - The first tensor as the numerator.
* **b** (tf.Tensor | TypedArray | Array) - The second tensor as the denominator. Must have the same dtype as `a`.

### Request Example
```javascript
const a = tf.tensor1d([1, 4, 9, 16]);
const b = tf.tensor1d([1, 2, 3, 4]);

a.div(b).print();  // or tf.div(a, b)
```

### Response
#### Success Response (200)
* **tf.Tensor**: The resulting tensor after division.

#### Response Example
```json
{
  "data": [1, 2, 3, 4],
  "shape": [4],
  "dtype": "float32"
}
```
```

--------------------------------

### tf.logicalXor (a, b)

Source: https://js.tensorflow.org/api/latest/index

Returns the truth value of `a XOR b` element-wise. Supports broadcasting.

```APIDOC
## POST /tf.logicalXor

### Description
Returns the truth value of `a XOR b` element-wise. Supports broadcasting.

### Method
POST

### Endpoint
/tf.logicalXor

### Parameters
#### Request Body
- **a** (tf.Tensor|TypedArray|Array) - Required - The first input tensor. Must be of dtype bool.
- **b** (tf.Tensor|TypedArray|Array) - Required - The second input tensor. Must be of dtype bool.

### Request Example
```json
{
  "a": [false, false, true, true],
  "b": [false, true, false, true]
}
```

### Response
#### Success Response (200)
- **result** (tf.Tensor) - The resulting tensor with truth values.

#### Response Example
```json
{
  "result": [false, true, true, false]
}
```
```

--------------------------------

### tf.logSumExp

Source: https://js.tensorflow.org/api/latest/index

Computes the log(sum(exp(elements across the reduction dimensions))) of a tensor.

```APIDOC
## tf.logSumExp

### Description
Computes the log(sum(exp(elements across the reduction dimensions))) of a tensor. Reduces the input along the dimensions given in `axis`. Unless `keepDims` is true, the rank of the array is reduced by 1 for each entry in `axis`. If `keepDims` is true, the reduced dimensions are retained with length 1. If `axis` has no entries, all dimensions are reduced, and an array with a single element is returned.

### Method
`tf.logSumExp(x, axis?, keepDims?)
`

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **x** (tf.Tensor|TypedArray|Array) - Required - The input tensor.
- **axis** (number|number[]) - Optional - The dimension(s) to reduce. If null (the default), reduces all dimensions.
- **keepDims** (boolean) - Optional - If true, retains reduced dimensions with length of 1. Defaults to false.

### Request Example
```javascript
const x = tf.tensor1d([1, 2, 3]);
x.logSumExp().print();  // or tf.logSumExp(x)

const y = tf.tensor2d([1, 2, 3, 4], [2, 2]);
const axis = 1;
y.logSumExp(axis).print();  // or tf.logSumExp(y, axis)
```

### Response
#### Success Response (200)
- **result** (tf.Tensor) - The tensor after reduction.

#### Response Example
```json
{
  "example": "tf.Tensor"
}
```
```

--------------------------------

### Compute Exponential Linear Element-wise with tf.elu

Source: https://js.tensorflow.org/api/latest/index

Computes the exponential linear element-wise function: `x > 0 ? x : (e ^ x) - 1`. It takes a tensor as input and returns a tensor with the computed values. Available for webgpu, webgl, wasm, and cpu backends.

```javascript
const x = tf.tensor1d([-1, 1, -3, 2]);

x.elu().print();  // or tf.elu(x)
```

--------------------------------

### Check for Finite Elements with tf.isFinite

Source: https://js.tensorflow.org/api/latest/index

Returns a boolean tensor indicating which elements of the input tensor are finite. It takes a tensor as input and returns a boolean tensor. Available for webgpu, webgl, wasm, and cpu backends.

```javascript
const x = tf.tensor1d([NaN, Infinity, -Infinity, 0, 1]);

x.isFinite().print();  // or tf.isNaN(x)
```

--------------------------------

### Tensor Contraction with Einsum

Source: https://js.tensorflow.org/api/latest/index

The `einsum` function allows for defining tensors by specifying their element-wise computation using Einstein summation notation. It supports various operations like matrix multiplication, dot products, batch dot products, outer products, and matrix transpositions.

```APIDOC
## POST /einsum

### Description
Performs tensor contraction over specified indices and outer product based on Einstein summation notation.

### Method
POST

### Endpoint
`/einsum`

### Parameters
#### Request Body
- **equation** (string) - Required - A string describing the contraction, in the same format as numpy.einsum.
- **tensors** (tf.Tensor[]) - Required - The input(s) to contract (each one a Tensor), whose shapes should be consistent with equation.

### Request Example
```json
{
  "equation": "ij,jk->ik",
  "tensors": [
    [[1, 2, 3], [4, 5, 6]],
    [[0, 1], [2, 3], [4, 5]]
  ]
}
```

### Response
#### Success Response (200)
- **result** (tf.Tensor) - The resulting tensor after the einsum operation.

#### Response Example
```json
{
  "result": [[30, 36, 42], [66, 81, 96]]
}
```

### Limitations
- Does not support >2 input tensors.
- Does not support duplicate axes for any given input tensor (e.g., 'ii->' is not supported).
- The `...` notation is not supported.
```

--------------------------------

### Compute Sine of Tensor Elements (JavaScript)

Source: https://js.tensorflow.org/api/latest/index

Computes the element-wise sine of a tensor. Supports CPU, WebGL, WebGPU, and WASM backends. Takes a tensor, TypedArray, or Array as input and returns a tensor.

```javascript
const x = tf.tensor1d([0, Math.PI / 2, Math.PI * 3 / 4]);

x.sin().print();  // or tf.sin(x)
```

--------------------------------

### tf.image.cropAndResize

Source: https://js.tensorflow.org/api/latest/index

Extracts crops from the input image tensor and resizes them using bilinear sampling or nearest neighbor sampling to a common output size.

```APIDOC
## POST /tf.image.cropAndResize

### Description
Extracts crops from the input image tensor and resizes them using bilinear sampling or nearest neighbor sampling to a common output size.

### Method
POST

### Endpoint
/tf.image.cropAndResize

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **image** (tf.Tensor4D|TypedArray|Array) - Required - 4D tensor of shape `[batch, imageHeight, imageWidth, depth]`, specifying the batch of images from which to take crops.
- **boxes** (tf.Tensor2D|TypedArray|Array) - Required - 2D float32 tensor of shape `[numBoxes, 4]`. Each entry is `[y1, x1, y2, x2]`, where `(y1, x1)` and `(y2, x2)` are the normalized coordinates of the box in the `boxInd[i]`th image in the batch.
- **boxInd** (tf.Tensor1D|TypedArray|Array) - Required - 1D int32 tensor of shape `[numBoxes]` with values in range `[0, batch)` that specifies the image that the `i`-th box refers to.
- **cropSize** ([number, number]) - Required - 1D int32 tensor of 2 elements `[cropHeigh, cropWidth]` specifying the size to which all crops are resized to.
- **method** ('bilinear'|'nearest') - Optional - String specifying the sampling method for resizing. Defaults to `'bilinear'`.
- **extrapolationValue** (number) - Optional - A threshold for deciding when to remove boxes based on score. Defaults to 0.

### Request Example
```json
{
  "image": "[[[[...]]]]",
  "boxes": [[0.0, 0.0, 1.0, 1.0]],
  "boxInd": [0],
  "cropSize": [100, 100],
  "method": "bilinear",
  "extrapolationValue": 0
}
```

### Response
#### Success Response (200)
- **output** (tf.Tensor4D) - The resulting tensor containing the cropped and resized images.

#### Response Example
```json
{
  "output": "[[[[...]]]]"
}
```
```

--------------------------------

### tf.asin

Source: https://js.tensorflow.org/api/latest/index

Computes the arc sine of a tensor element-wise.

```APIDOC
## tf.asin (x)

### Description
Computes asin of the input tf.Tensor element-wise: `asin(x)`.

### Method
`tf.asin` or Tensor.asin

### Parameters
* `x` (tf.Tensor|TypedArray|Array) - The input tensor.

### Request Example
```javascript
const x = tf.tensor1d([0, 1, -1, .7]);

x.asin().print();  // or tf.asin(x)
```

### Response Example
```javascript
// Tensor
//   [0,
//    1.5707963,
//   -1.5707963,
//    0.7753975]
```
```

--------------------------------

### tf.layers.dense

Source: https://js.tensorflow.org/api/latest/index

Creates a dense (fully connected) layer.

```APIDOC
## tf.layers.dense

### Description
Creates a dense (fully connected) layer. This layer implements the operation: `output = activation(dot(input, kernel) + bias)`.

### Method
`tf.layers.dense(args)`

### Endpoint
N/A (Client-side API)

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
None

* **args** (Object) - Configuration options for the layer.
  * **units** (number) - Positive integer, dimensionality of the output space.
  * **activation** ('elu'|'hardSigmoid'|'linear'|'relu'|'relu6'| 'selu'|'sigmoid'|'softmax'|'softplus'|'softsign'|'tanh'|'swish'|'mish'|'gelu'|'gelu_new') - Activation function to use. If unspecified, no activation is applied.
  * **useBias** (boolean) - Whether to apply a bias.
  * **kernelInitializer** ('constant'|'glorotNormal'|'glorotUniform'|'heNormal'|'heUniform'|'identity'| 'leCunNormal'|'leCunUniform'|'ones'|'orthogonal'|'randomNormal'| 'randomUniform'|'truncatedNormal'|'varianceScaling'|'zeros'|string|tf.initializers.Initializer) - Initializer for the dense kernel weights matrix.
  * **biasInitializer** ('constant'|'glorotNormal'|'glorotUniform'|'heNormal'|'heUniform'|'identity'| 'leCunNormal'|'leCunUniform'|'ones'|'orthogonal'|'randomNormal'| 'randomUniform'|'truncatedNormal'|'varianceScaling'|'zeros'|string|tf.initializers.Initializer) - Initializer for the bias vector.
  * **inputDim** (number) - If specified, defines `inputShape` as `[inputDim]`.
  * **kernelConstraint** ('maxNorm'|'minMaxNorm'|'nonNeg'|'unitNorm'|string|tf.constraints.Constraint) - Constraint for the kernel weights.
  * **biasConstraint** ('maxNorm'|'minMaxNorm'|'nonNeg'|'unitNorm'|string|tf.constraints.Constraint) - Constraint for the bias vector.
  * **kernelRegularizer** ('l1l2'|string|Regularizer) - Regularizer function applied to the dense kernel weights matrix.
  * **biasRegularizer** ('l1l2'|string|Regularizer) - Regularizer function applied to the bias vector.
  * **activityRegularizer** ('l1l2'|string|Regularizer) - Regularizer function applied to the activation.
  * **inputShape** ((null | number)[]) - If defined, will be used to create an input layer. Only applicable to input layers.
  * **batchInputShape** ((null | number)[]) - If defined, will be used to create an input layer. Only applicable to input layers.

### Request Example
```json
{
  "units": 10,
  "activation": "relu",
  "inputShape": [5]
}
```

### Response
#### Success Response (200)
Dense layer instance.

#### Response Example
```json
{
  "denseLayer": "<tf.layers.Layer object>"
}
```
```

--------------------------------

### Time Function Execution with tf.time

Source: https://js.tensorflow.org/api/latest/index

Executes a function and returns a promise that resolves with timing information, including kernelMs and wallMs. For WebGL, it also provides uploadWaitMs and downloadWaitMs.

```javascript
const x = tf.randomNormal([20, 20]);
const time = await tf.time(() => x.matMul(x));

console.log(`kernelMs: ${time.kernelMs}, wallTimeMs: ${time.wallMs}`);

```

--------------------------------

### Compute Floor Element-wise with tf.floor

Source: https://js.tensorflow.org/api/latest/index

Computes the floor of the input tensor element-wise (`floor(x)`). It takes a tensor as input and returns a tensor with the computed values. Available for webgpu, webgl, wasm, and cpu backends.

```javascript
const x = tf.tensor1d([.6, 1.1, -3.3]);

x.floor().print();  // or tf.floor(x)
```

--------------------------------

### tf.linalg.bandPart

Source: https://js.tensorflow.org/api/latest/index

Copy a tensor setting everything outside a central band in each innermost matrix to zero.

```APIDOC
## POST /tf.linalg.bandPart

### Description
Copy a tensor setting everything outside a central band in each innermost matrix to zero. The band part is computed as follows: Assume input has `k` dimensions `[I, J, K, ..., M, N]`, then the output is a tensor with the same shape where `band[i, j, k, ..., m, n] = in_band(m, n) * input[i, j, k, ..., m, n]`. The indicator function `in_band(m, n) = (num_lower < 0 || (m-n) <= num_lower)` `&& (num_upper < 0 || (n-m) <= num_upper)`

### Method
POST

### Endpoint
/tf.linalg.bandPart

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **a** (tf.Tensor|TypedArray|Array) - Required - The input tensor.
- **numLower** (number|tf.Scalar) - Required - Number of subdiagonals to keep. If negative, keep entire lower triangle.
- **numUpper** (number|tf.Scalar) - Required - Number of subdiagonals to keep. If negative, keep entire upper triangle.

### Request Example
```json
{
  "a": [[ 0,  1,  2, 3],
         [-1,  0,  1, 2],
         [-2, -1,  0, 1],
         [-3, -2, -1, 0]],
  "numLower": 1,
  "numUpper": -1
}
```

### Response
#### Success Response (200)
- **tensor** (tf.Tensor) - The tensor with the band part copied.

#### Response Example
```json
{
  "tensor": "[Tensor object representation]"
}
```
```

--------------------------------

### Apply General Activation Function in TensorFlow.js

Source: https://js.tensorflow.org/api/latest/index

Applies a specified activation function element-wise to the output of a preceding layer. This is useful for isolating activation computations or applying activations not directly available as layer types. It requires the `activation` parameter.

```javascript
const input = tf.input({shape: [5]});
const activationLayer = tf.layers.activation({activation: 'relu6'});
const output = activationLayer.apply(input);
console.log(output);
```

--------------------------------

### Construct tf.AdagradOptimizer in JavaScript

Source: https://js.tensorflow.org/api/latest/index

Constructs a tf.AdagradOptimizer that uses the Adagrad algorithm. Adagrad adapts the learning rate to the parameters, performing larger updates for sparse parameters and smaller updates for frequent parameters.

```javascript
const learningRate = 0.01;
const optimizer = tf.train.adagrad(learningRate);

```

--------------------------------

### skip (count)

Source: https://js.tensorflow.org/api/latest/index

Creates a `Dataset` that skips `count` initial elements from this dataset.

```APIDOC
## skip (count)

### Description
Creates a `Dataset` that skips `count` initial elements from this dataset.

### Method
skip

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **count** (number) - The number of elements of this dataset that should be skipped to form the new dataset. If `count` is greater than the size of this dataset, the new dataset will contain no elements. If `count` is `undefined` or negative, skips the entire dataset.

### Request Example
```javascript
const a = tf.data.array([1, 2, 3, 4, 5, 6]).skip(3);
a.forEachAsync(e => console.log(e));
```

### Response
#### Success Response (200)
- **tf.data.Dataset** - The dataset with skipped elements.

#### Response Example
```json
{
  "example": "Dataset object"
}
```
```

--------------------------------

### Compute Square Root of Tensor Elements (JavaScript)

Source: https://js.tensorflow.org/api/latest/index

Computes the element-wise square root of a tensor. Supports CPU, WebGL, WebGPU, and WASM backends. Accepts a tensor, TypedArray, or Array as input and returns a tensor.

```javascript
const x = tf.tensor1d([1, 2, 4, -1]);

x.sqrt().print();  // or tf.sqrt(x)
```

--------------------------------

### Compute Scaled Exponential Linear Unit (SELU) of Tensor Elements (JavaScript)

Source: https://js.tensorflow.org/api/latest/index

Computes the element-wise scaled exponential linear unit (SELU) of a tensor. For x < 0, it calculates scale * alpha * (exp(x) - 1), otherwise scale * x. Supports CPU, WebGL, WebGPU, and WASM backends. Accepts a tensor, TypedArray, or Array and returns a tensor.

```javascript
const x = tf.tensor1d([-1, 2, -3, 4]);

x.selu().print();  // or tf.selu(x)
```

--------------------------------

### tf.tensor5d

Source: https://js.tensorflow.org/api/latest/index

Creates a rank-5 tf.Tensor with the provided values, shape and dtype. Recommended for readability over tf.tensor().

```APIDOC
## tf.tensor5d

### Description
Creates a rank-5 tf.Tensor with the provided values, shape and dtype. This function is recommended for code readability compared to using the generic `tf.tensor()`.

### Method
`tf.tensor5d(values, shape?, dtype?): tf.Tensor5D`

### Parameters
* **values** (TypedArray|Array) - The values of the tensor. Can be a nested array of numbers, a flat array, or a TypedArray.
* **shape** ([number, number, number, number, number]) - The shape of the tensor. Optional. If not provided, it is inferred from `values`.
* **dtype** ('float32'|'int32'|'bool'|'complex64'|'string') - The data type of the tensor. Optional.

### Request Example
```javascript
// Pass a nested array.
tf.tensor5d([[[[[1],[2]],[[3],[4]]],[[[5],[6]],[[7],[8]]]]]).print();

// Pass a flat array and specify a shape.
tf.tensor5d([1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 2, 2, 1]).print();
```

### Response Example
```json
{
  "example": "tf.Tensor5D"
}
```
```

--------------------------------

### tf.zeros

Source: https://js.tensorflow.org/api/latest/index

Creates a tf.Tensor with all elements set to 0.

```APIDOC
## POST /tf.zeros

### Description
Creates a tf.Tensor with all elements set to 0.

### Method
POST

### Endpoint
/tf.zeros

### Parameters
#### Query Parameters
- **shape** (number[]) - Required - An array of integers defining the output tensor shape.
- **dtype** ('float32'|'int32'|'bool'|'complex64'|'string') - Optional - The type of an element in the resulting tensor. Can be 'float32', 'int32' or 'bool'. Defaults to 'float'.

### Request Example
```json
{
  "shape": [2, 2]
}
```

### Response
#### Success Response (200)
- **tf.Tensor** - A tensor with all elements set to 0.

#### Response Example
```json
{
  "tensor": [[0, 0], [0, 0]]
}
```
```

--------------------------------

### tf.oneHot

Source: https://js.tensorflow.org/api/latest/index

Creates a one-hot encoded tensor.

```APIDOC
## POST /websites/js_tensorflow_api/tf.oneHot

### Description
Creates a one-hot tf.Tensor. The locations represented by `indices` take value `onValue` (defaults to 1), while all other locations take value `offValue` (defaults to 0). If `indices` is rank `R`, the output has rank `R+1` with the last axis of size `depth`. `indices` used to encode prediction class must start from 0. For example, if you have 3 classes of data, class 1 should be encoded as 0, class 2 should be 1, and class 3 should be 2.

### Method
POST

### Endpoint
/websites/js_tensorflow_api/tf.oneHot

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **indices** (tf.Tensor|TypedArray|Array) - tf.Tensor of indices with dtype `int32`. Indices must start from 0.
- **depth** (number) - The depth of the one hot dimension.
- **onValue** (number) - A number used to fill in the output when the index matches the location. Optional
- **offValue** (number) - A number used to fill in the output when the index does not match the location. Optional
- **dtype** ('float32'|'int32'|'bool'|'complex64'|'string') - The dtype of the output tensor, default to 'int32'. Optional

### Request Example
```json
{
  "indices": [0, 1],
  "depth": 3
}
```

### Response
#### Success Response (200)
- **one_hot_tensor** (tf.Tensor) - The one-hot encoded tensor.

#### Response Example
```json
{
  "one_hot_tensor": [[1, 0, 0], [0, 1, 0]]
}
```
```

--------------------------------

### Tile Tensor1D using TensorFlow.js

Source: https://js.tensorflow.org/api/latest/index

Repeats a 1-dimensional tensor multiple times to create a larger tensor. The `tile` operation in TensorFlow.js replicates the input tensor based on the `reps` parameter. This is useful for broadcasting and expanding tensor dimensions. Works across different backends.

```javascript
const a = tf.tensor1d([1, 2]);

a.tile([2]).print();    // or tf.tile(a, [2])

```

--------------------------------

### MaxPooling3D API

Source: https://js.tensorflow.org/api/latest/index

Provides max pooling for 3D spatial data. It reduces the spatial dimensions of the input by taking the maximum value over a window.

```APIDOC
## POST /tf.layers.maxPooling3d

### Description
Max pooling operation for 3D data. It reduces the spatial dimensions of the input by taking the maximum value over a window.

### Method
POST

### Endpoint
/tf.layers.maxPooling3d

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **args** (Object) - Required - Configuration object for the pooling layer.
  - **poolSize** (number | [number, number, number]) - Optional - Factors by which to downscale in each dimension [depth, height, width]. Defaults to the layer's default.
  - **strides** (number | [number, number, number]) - Optional - The size of the stride in each dimension of the pooling window. If null, defaults to `poolSize`. Defaults to the layer's default.
  - **padding** ('valid' | 'same' | 'causal') - Optional - The padding type to use for the pooling layer. Defaults to 'valid'.
  - **dataFormat** ('channelsFirst' | 'channelsLast') - Optional - The data format to use for the pooling layer. Defaults to 'channelsLast'.
  - **inputShape** (Array<number | null>) - Optional - If defined, will be used to create an input layer to insert before this layer. This argument is only applicable to input layers.
  - **batchInputShape** (Array<number | null>) - Optional - If defined, will be used to create an input layer to insert before this layer. If both `inputShape` and `batchInputShape` are defined, `batchInputShape` will be used. This argument is only applicable to input layers.
  - **batchSize** (number) - Optional - If `inputShape` is specified and `batchInputShape` is _not_ specified, `batchSize` is used to construct the `batchInputShape`.
  - **dtype** ('float32' | 'int32' | 'bool' | 'complex64' | 'string') - Optional - The data-type for this layer. Defaults to 'float32'. This argument is only applicable to input layers.
  - **name** (string) - Optional - Name for this layer.
  - **trainable** (boolean) - Optional - Whether the weights of this layer are updatable by `fit`. Defaults to true.
  - **weights** (Array<tf.Tensor>) - Optional - Initial weight values of the layer.
  - **inputDType** ('float32' | 'int32' | 'bool' | 'complex64' | 'string') - Optional - Legacy support. Do not use for new code.

### Request Example
```json
{
  "args": {
    "poolSize": [2, 2, 2],
    "strides": [2, 2, 2],
    "padding": "same",
    "dataFormat": "channelsFirst"
  }
}
```

### Response
#### Success Response (200)
- **MaxPooling3D** (Object) - An instance of the MaxPooling3D layer.

#### Response Example
```json
{
  "layerType": "MaxPooling3D",
  "name": "maxPooling3d_1"
}
```
```

--------------------------------

### TensorFlow.js: Batch Matrix Transpose using einsum

Source: https://js.tensorflow.org/api/latest/index

Transposes each matrix within a batch of 3D tensors using `einsum`. The equation 'bij->bji' swaps the second and third dimensions for each batch element 'b'.

```javascript
const x = tf.tensor3d([[[1, 2], [3, 4]], [[-1, -2], [-3, -4]]]);
x.print();
tf.einsum('bij->bji', x).print();
```

--------------------------------

### Create Tensor Buffer and Convert to Tensor - tf.buffer

Source: https://js.tensorflow.org/api/latest/index

Initializes an empty tf.TensorBuffer with a specified shape and dtype, allowing values to be set at specific indices before conversion to an immutable tensor.

```javascript
// Create a buffer and set values at particular indices.
const buffer = tf.buffer([2, 2]);
buffer.set(3, 0, 0);
buffer.set(5, 1, 0);

// Convert the buffer back to a tensor.
buffer.toTensor().print();

```

--------------------------------

### Hash Strings to Buckets with tf.stringToHashBucketFast

Source: https://js.tensorflow.org/api/latest/index

Converts each string in an input tensor to its hash modulo a specified number of buckets. The hash function is deterministic within a process but not cryptographically secure. Suitable for scenarios where CPU time is limited and inputs are trusted.

```javascript
const result = tf.string.stringToHashBucketFast(
   ['Hello', 'TensorFlow', '2.x'], 3);
result.print(); // [0, 2, 2]
```

--------------------------------

### Define Model Input with tf.input() in JavaScript

Source: https://js.tensorflow.org/api/latest/index

Instantiates a symbolic input tensor for a model. This factory function ensures consistency with other TensorFlow.js generator functions. It's primarily used when constructing models with tf.model(), whereas tf.sequential() uses inputShape or inputLayer.

```javascript
const x = tf.input({shape: [32]});
const y = tf.layers.dense({units: 3, activation: 'softmax'}).apply(x);
const model = tf.model({inputs: x, outputs: y});
model.predict(tf.ones([2, 32])).print();

```

--------------------------------

### Compute Negation Element-wise with tf.neg

Source: https://js.tensorflow.org/api/latest/index

Computes the negation of the input tensor element-wise (`-1 * x`). It takes a tensor as input and returns a tensor with the computed values. Available for webgpu, webgl, wasm, and cpu backends.

```javascript
const x = tf.tensor2d([1, 2, -2, 0], [2, 2]);

x.neg().print();  // or tf.neg(x)
```

--------------------------------

### Compute General Norm with TensorFlow.js

Source: https://js.tensorflow.org/api/latest/index

Computes various norms (1-norm, 2-norm, inf-norm, p-norm, Frobenius norm) for scalar, vectors, and matrices using the norm function. Users can specify the order of the norm, the axis, and whether to keep the dimensions. Available for webgpu, webgl, wasm, and cpu.

```javascript
const x = tf.tensor1d([1, 2, 3, 4]);

x.norm().print();  // or tf.norm(x)
```

--------------------------------

### tf.stack

Source: https://js.tensorflow.org/api/latest/index

Stacks a list of tensors into one tensor of higher rank.

```APIDOC
## tf.stack

### Description
Stacks a list of rank-`R` tf.Tensors into one rank-`(R+1)` tf.Tensor.

### Method
`stack(tensors: Array<tf.Tensor | tf.TypedArray | tf.TensorLike>, axis?: number): tf.Tensor`

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
None

### Request Example
```javascript
const a = tf.tensor1d([1, 2]);
const b = tf.tensor1d([3, 4]);
const c = tf.tensor1d([5, 6]);
tf.stack([a, b, c]).print();
```

### Response
#### Success Response (200)
- **tf.Tensor** - The stacked tensor.

#### Response Example
```json
{
  "example": "tensor data"
}
```
```

--------------------------------

### Compute element-wise inverse hyperbolic sine using TensorFlow.js

Source: https://js.tensorflow.org/api/latest/index

Computes the inverse hyperbolic sine (asinh) of the input tensor element-wise. The `asinh` function takes a tensor `x` and returns a new tensor with the inverse hyperbolic sine values.

```javascript
const x = tf.tensor1d([0, 1, -1, .7]);

x.asinh().print();  // or tf.asinh(x)
```

--------------------------------

### tf.train.adam

Source: https://js.tensorflow.org/api/latest/index

Constructs a `tf.AdamOptimizer` that uses the Adam algorithm.

```APIDOC
## tf.train.adam

### Description
Constructs a `tf.AdamOptimizer` that uses the Adam algorithm.

### Method
`tf.train.adam`

### Endpoint
N/A (This is a function call within the library)

### Parameters
#### Path Parameters
N/A

#### Query Parameters
N/A

#### Request Body
N/A

### Request Example
```javascript
const learningRate = 0.01;
const optimizer = tf.train.adam(learningRate);
```

### Response
#### Success Response (200)
Returns a `tf.AdamOptimizer` instance.

#### Response Example
```json
{
  "optimizerType": "adam"
}
```
```

--------------------------------

### Bitwise AND Operation

Source: https://js.tensorflow.org/api/latest/index

Performs a bitwise AND operation on two input tensors.

```APIDOC
## tf.bitwiseAnd

### Description
Bitwise `AND` operation for input tensors. Given two input tensors, returns a new tensor with the `AND` calculated values. The method supports int32 values.

### Method
Not specified (assumed to be a function call)

### Endpoint
N/A

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **x** (tf.Tensor) - Required - The input tensor to be calculated.
- **y** (tf.Tensor) - Required - The input tensor to be calculated.

### Request Example
```javascript
const x = tf.tensor1d([0, 5, 3, 14], 'int32');
const y = tf.tensor1d([5, 0, 7, 11], 'int32');
tf.bitwiseAnd(x, y).print();
// Output: 
// Tensor
//    [0,
//     0,
//     3,
//     10]
```

### Response
#### Success Response (200)
- **result** (tf.Tensor) - A new tensor with the bitwise AND calculated values.
```

--------------------------------

### Enable Production Mode in TensorFlow.js

Source: https://js.tensorflow.org/api/latest/index

Enables production mode, which prioritizes performance by disabling correctness checks. This mode is recommended for production environments.

```javascript
tf.enableProdMode ();
```

--------------------------------

### tf.train.adagrad

Source: https://js.tensorflow.org/api/latest/index

Constructs a tf.AdagradOptimizer that uses the Adagrad algorithm.

```APIDOC
## tf.train.adagrad

### Description
Constructs a tf.AdagradOptimizer that uses the Adagrad algorithm.

### Method
`tf.train.adagrad`

### Endpoint
N/A (This is a function call within the library)

### Parameters
#### Path Parameters
N/A

#### Query Parameters
N/A

#### Request Body
N/A

### Request Example
```javascript
const learningRate = 0.01;
const optimizer = tf.train.adagrad(learningRate);
```

### Response
#### Success Response (200)
Returns a `tf.AdagradOptimizer` instance.

#### Response Example
```json
{
  "optimizerType": "adagrad"
}
```
```

--------------------------------

### Compute element-wise arcsine using TensorFlow.js

Source: https://js.tensorflow.org/api/latest/index

Computes the arcsine (inverse sine) of the input tensor element-wise. The `asin` function takes a tensor `x` and returns a new tensor with the arcsine values.

```javascript
const x = tf.tensor1d([0, 1, -1, .7]);

x.asin().print();  // or tf.asin(x)
```

--------------------------------

### Compute Sign of Tensor Elements (JavaScript)

Source: https://js.tensorflow.org/api/latest/index

Returns an element-wise indication of the sign of each number in the tensor. Supports CPU, WebGL, WebGPU, and WASM backends. Accepts a tensor, TypedArray, or Array and returns a tensor.

```javascript
const x = tf.tensor1d([.6, 1.1, -3.3, NaN, 0]);

x.sign().print();  // or tf.sign(x)
```

--------------------------------

### Compute Sigmoid of Tensor Elements (JavaScript)

Source: https://js.tensorflow.org/api/latest/index

Computes the element-wise sigmoid function of a tensor, defined as 1 / (1 + exp(-x)). Supports CPU, WebGL, WebGPU, and WASM backends. Takes a tensor, TypedArray, or Array as input and returns a tensor.

```javascript
const x = tf.tensor1d([0, -1, 2, -3]);

x.sigmoid().print();  // or tf.sigmoid(x)
```

--------------------------------

### Advanced 1D Convolutional Layer Configuration in TensorFlow.js

Source: https://js.tensorflow.org/api/latest/index

This snippet illustrates an advanced configuration of tf.layers.conv1d, including data format, dilation rate, and kernel/bias initializers. These parameters allow for fine-tuning the convolution operation for complex tasks.

```javascript
const advancedConvLayer = tf.layers.conv1d({
  filters: 128,
  kernelSize: [3, 3], // 2D kernel size for a 1D input
  strides: 1,
  padding: 'causal',
  dataFormat: 'channelsLast',
  dilationRate: 2,
  activation: 'sigmoid',
  useBias: false,
  kernelInitializer: 'glorotUniform',
  biasInitializer: 'zeros',
  kernelConstraint: 'maxNorm',
  biasRegularizer: 'l1',
  inputShape: [20, 1] // Sequences of 20 time steps with 1 channel
});
console.log(advancedConvLayer);
```

--------------------------------

### Compute Gradients of a Function with tf.grads

Source: https://js.tensorflow.org/api/latest/index

Computes the gradients of a function `f` with respect to its inputs. The function `f` must accept one or more tensors and return a single tensor. If `f` takes a single input, `tf.grad()` is recommended. This function is available for webgpu, webgl, wasm, and cpu backends.

```javascript
const f = (a, b) => a.mul(b);
const g = tf.grads(f);

const a = tf.tensor1d([2, 3]);
const b = tf.tensor1d([-2, -3]);
const [da, db] = g([a, b]);
console.log('da');
da.print();
console.log('db');
db.print();

```

--------------------------------

### Model Prediction: predict

Source: https://js.tensorflow.org/api/latest/index

Generates output predictions for input samples. Computation is done in batches. Note: the 'step' mode of predict() is currently not supported.

```APIDOC
## predict

### Description
Generates output predictions for the input samples. Computation is done in batches.

### Method
`predict(x, args?)

### Parameters
#### Path Parameters
* x (tf.Tensor|tf.Tensor[]) - Required - The input data, as a Tensor, or an `Array` of tf.Tensors if the model has multiple inputs.
* args (Object) - Optional - A `ModelPredictArgs` object containing optional fields.
  * batchSize (number) - Optional - Batch size (Integer). If unspecified, it will default to 32.
  * verbose (boolean) - Optional - Verbosity mode. Defaults to false.

### Request Example
```javascript
const model = tf.sequential({
   layers: [tf.layers.dense({units: 1, inputShape: [10]})]
});
model.predict(tf.ones([8, 10]), {batchSize: 4}).print();
```

### Returns
tf.Tensor|tf.Tensor[]

### Notes
- The "step" mode of predict() is currently not supported. This is because the TensorFlow.js core backend is imperative only.
- Supported backends: webgpu, webgl, wasm, cpu.
```

--------------------------------

### Compute Rounding of Tensor Elements (JavaScript)

Source: https://js.tensorflow.org/api/latest/index

Computes the element-wise rounding of a tensor using banker's rounding. Supports CPU, WebGL, WebGPU, and WASM backends. Accepts a tensor, TypedArray, or Array as input and returns a tensor.

```javascript
const x = tf.tensor1d([.6, 1.1, -3.3]);

x.round().print();  // or tf.round(x)
```

--------------------------------

### Create and Assign Value to a Variable

Source: https://js.tensorflow.org/api/latest/index

Creates a tf.Variable with an initial tensor value and allows for subsequent assignment of new tensor values. The variable can optionally be marked as trainable and given a name. Supports multiple backends.

```javascript
const x = tf.variable(tf.tensor([1, 2, 3]));
x.assign(tf.tensor([4, 5, 6]));
x.print();
```

--------------------------------

### Load Graph Model from URL with tf.loadGraphModel() in JavaScript

Source: https://js.tensorflow.org/api/latest/index

Loads a graph model from a provided URL. This function is asynchronous and returns a Promise that resolves to the loaded model. It's suitable for loading models saved in the SavedModel format. Options can be provided for fetching and loading behavior.

```javascript
const modelUrl = 'https://storage.googleapis.com/tfjs-models/savedmodel/mobilenet_v2_1.0_224/model.json';
const model = await tf.loadGraphModel(modelUrl);
const zeros = tf.zeros([1, 224, 224, 3]);
model.predict(zeros).print();

```

```javascript
const modelUrl = 'https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/classification/2';
const model = await tf.loadGraphModel(modelUrl, {fromTFHub: true});
const zeros = tf.zeros([1, 224, 224, 3]);
model.predict(zeros).print();

```

--------------------------------

### Compute Exponential Element-wise with tf.exp

Source: https://js.tensorflow.org/api/latest/index

Computes the exponential of the input tensor element-wise (`e ^ x`). It takes a tensor as input and returns a tensor with the computed values. Available for webgpu, webgl, wasm, and cpu backends.

```javascript
const x = tf.tensor1d([1, 2, -3]);

x.exp().print();  // or tf.exp(x)
```

--------------------------------

### TensorFlow.js: Outer Product using einsum

Source: https://js.tensorflow.org/api/latest/index

Calculates the outer product of two 1D tensors using `einsum`. The equation 'i,j->ij' indicates that each element 'i' of the first tensor is paired with each element 'j' of the second tensor to form a 2D result.

```javascript
const x = tf.tensor1d([1, 3, 5]);
const y = tf.tensor1d([2, 4, 6]);
x.print();
y.print();
tf.einsum('i,j->ij', x, y).print();
```

--------------------------------

### Capture Image from Webcam using tf.data.webcam

Source: https://js.tensorflow.org/api/latest/index

Creates an iterator to generate TensorFlow.js Tensors from a webcam video stream. This function requires a browser environment and a connected webcam. It will request camera permissions upon execution. The captured image can then be processed.

```javascript
const videoElement = document.createElement('video');
videoElement.width = 100;
videoElement.height = 100;
const cam = await tf.data.webcam(videoElement);
const img = await cam.capture();
img.print();
cam.stop();

```

--------------------------------

### Calculate Bincount of Tensor Elements

Source: https://js.tensorflow.org/api/latest/index

Outputs a vector with a specified size, representing the bincount of the input tensor. The output vector has the same dtype as the weights input. This function is used for counting occurrences of values.

```javascript
tf.bincount(x, weights, size)
```

--------------------------------

### TensorFlow.js: Gamma Distribution Sampling

Source: https://js.tensorflow.org/api/latest/index

Creates a tensor with values sampled from a gamma distribution using `tf.randomGamma`. It requires shape and alpha parameters, with optional beta, dtype, and seed.

```javascript
tf.randomGamma([2, 2], 1).print();
```

--------------------------------

### Clip tensor values to a specified range using TensorFlow.js

Source: https://js.tensorflow.org/api/latest/index

Clips tensor values element-wise to be within a specified minimum and maximum value. The `clipByValue` function takes a tensor `x` and two numbers, `clipValueMin` and `clipValueMax`, and returns a new tensor with values clipped to this range.

```javascript
const x = tf.tensor1d([-1, 2, -3, 4]);

x.clipByValue(-2, 3).print();  // or tf.clipByValue(x, -2, 3)
```

--------------------------------

### Replace Regex Matches in Strings with TensorFlow.js

Source: https://js.tensorflow.org/api/latest/index

Replaces occurrences of a pattern within input strings using a rewrite string. It accepts a string tensor, a pattern, and a rewrite rule. The replacement can be global or only affect the first match. Supported on WebGPU, WebGL, WASM, and CPU.

```javascript
const result = tf.string.staticRegexReplace(
     ['format       this   spacing      better'], ' +', ' ');
result.print(); // ['format this spacing better']
```

--------------------------------

### tf.eye

Source: https://js.tensorflow.org/api/latest/index

Creates an identity matrix.

```APIDOC
## POST /websites/js_tensorflow_api/tf.eye

### Description
Creates an identity matrix.

### Method
POST

### Endpoint
/websites/js_tensorflow_api/tf.eye

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **numRows** (number) - Number of rows.
- **numColumns** (number) - Number of columns. Defaults to `numRows`. Optional
- **batchShape** ([ number ]|[number, number]|[number, number, number]|[number, number, number, number]) - If provided, will add the batch shape to the beginning of the shape of the returned tf.Tensor by repeating the identity matrix. Optional
- **dtype** ('float32'|'int32'|'bool'|'complex64'|'string') - Data type. Optional

### Request Example
```json
{
  "numRows": 3,
  "numColumns": 4,
  "batchShape": [2, 3],
  "dtype": "float32"
}
```

### Response
#### Success Response (200)
- **identity_matrix** (tf.Tensor2D) - An identity matrix.

#### Response Example
```json
{
  "identity_matrix": [[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]]
}
```
```

--------------------------------

### tf.initializers.ones

Source: https://js.tensorflow.org/api/latest/index

Initializer that generates tensors filled with the value 1.

```APIDOC
## tf.initializers.ones ()

### Description
Initializer that generates tensors initialized to 1.

### Method
Function

### Endpoint
N/A (Client-side JavaScript function)

### Parameters
None

### Request Example
```json
{}
```

### Response
#### Success Response (tf.initializers.Initializer)
Returns an instance of tf.initializers.Initializer.

#### Response Example
```json
{
  "instance": "tf.initializers.Initializer"
}
```
```

--------------------------------

### Compute Dropout using TensorFlow.js

Source: https://js.tensorflow.org/api/latest/index

Computes dropout, a regularization technique that randomly sets elements of the input tensor to zero during training. This function is stateless and requires the caller to manage the step count if needed. It supports various backends like WebGPU, WebGL, WASM, and CPU.

```javascript
const x = tf.tensor1d([1, 2, 2, 1]);
const rate = 0.75;
const output = tf.dropout(x, rate);
output.print();
```

--------------------------------

### GRU Cell Call Method Signature

Source: https://js.tensorflow.org/api/latest/index

The `call()` method of a GRU cell accepts input tensors and optionally cell states and constants. It returns the output at time t and the cell states at time t+1. The signature is flexible to handle various input and state configurations.

```typescript
call(inputs: [Tensor, Tensor] | [Tensor, Tensor, Tensor[]], constants?: Tensor[]): [Tensor, Tensor[]]
```

--------------------------------

### tf.tensor

Source: https://js.tensorflow.org/api/latest/index

Creates a tensor from various data types including TypedArrays, nested arrays, WebGLData, and WebGPUData.

```APIDOC
## POST /api/v1/tensor

### Description
Creates a tensor from various data types including TypedArrays, nested arrays, WebGLData, and WebGPUData.

### Method
POST

### Endpoint
/api/v1/tensor

### Parameters
#### Request Body
- **values** (TypedArray|Array|WebGLData|WebGPUData) - Required - The values of the tensor. Can be nested array of numbers, or a flat array, or a TypedArray(At the moment it supports Uint8Array, Uint8ClampedArray, Int32Array, Float32Array) data types, or a `WebGLData` object, or a `WebGPUData` object. If the values are strings, they will be encoded as utf-8 and kept as `Uint8Array[]`. If the values is a `WebGLData` object, the dtype could only be 'float32' or 'int32' and the object has to have: 1. texture, a `WebGLTexture`, the texture must share the same `WebGLRenderingContext` with TFJS's WebGL backend (you could create a custom WebGL backend from your texture's canvas) and the internal texture format for the input texture must be floating point or normalized integer; 2. height, the height of the texture; 3. width, the width of the texture; 4. channels, a non-empty subset of 'RGBA', indicating the values of which channels will be passed to the tensor, such as 'R' or 'BR' (The order of the channels affect the order of tensor values. ). (If the values passed from texture is less than the tensor size, zeros will be padded at the rear.). If the values is a `WebGPUData` object, the dtype could only be 'float32' or 'int32 and the object has to have: buffer, a `GPUBuffer`. The buffer must:
    1. share the same `GPUDevice` with TFJS's WebGPU backend; 2. buffer.usage should at least support GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC; 3. buffer.size should not be smaller than the byte size of tensor shape. WebGPUData optionally supports zero copy by flag zeroCopy. When zeroCopy is false or undefined(default),this passing GPUBuffer can be destroyed after tensor is created. When zeroCopy is true, this GPUBuffer is bound directly by the tensor, so do not destroy this GPUBuffer until all access is done.
- **shape** (number[]) - Optional - The shape of the tensor. Optional. If not provided, it is inferred from `values`. Optional 
- **dtype** ('float32'|'int32'|'bool'|'complex64'|'string') - Optional - The data type. Optional 

### Response
#### Success Response (200)
- **tensor** (tf.Tensor) - The created tensor.

#### Response Example
```json
{
  "tensor": "<Tensor Object>"
}
```
```

--------------------------------

### Compute Reciprocal of Tensor Elements (JavaScript)

Source: https://js.tensorflow.org/api/latest/index

Calculates the element-wise reciprocal of a given tensor. Supports CPU, WebGL, WebGPU, and WASM backends. Takes a tensor, TypedArray, or Array as input and returns a tensor.

```javascript
const x = tf.tensor1d([0, 1, 2]);

x.reciprocal().print();  // or tf.reciprocal(x)
```

--------------------------------

### Skip Initial Elements from Dataset with JavaScript

Source: https://js.tensorflow.org/api/latest/index

The `skip` method creates a new dataset that excludes a specified number of initial elements from the original dataset. If the count exceeds the dataset size, the new dataset will be empty. Skipping an undefined or negative count skips the entire dataset.

```javascript
const a = tf.data.array([1, 2, 3, 4, 5, 6]).skip(3);
await a.forEachAsync(e => console.log(e));

```

--------------------------------

### Compute Reciprocal Square Root of Tensor Elements (JavaScript)

Source: https://js.tensorflow.org/api/latest/index

Calculates the element-wise reciprocal square root of a tensor (1 / sqrt(x)). Supports CPU, WebGL, WebGPU, and WASM backends. Takes a tensor, TypedArray, or Array as input and returns a tensor.

```javascript
const x = tf.tensor1d([1, 2, 4, -1]);

x.rsqrt().print();  // or tf.rsqrt(x)
```

--------------------------------

### tf.denseBincount

Source: https://js.tensorflow.org/api/latest/index

Computes a histogram of a sparse tensor by binning values in `x` into `size` bins. Supports optional weights.

```APIDOC
## tf.denseBincount

### Description
Computes a histogram of a sparse tensor by binning values in `x` into `size` bins. If `weights` are empty, then index `i` stores the number of times the value `i` is counted in `x`. If `weights` are non-empty, then index `i` stores the sum of the value in `weights` at each index where the corresponding value in `x` is `i`. Values in `x` outside of the range [0, size) are ignored.

### Method
`tf.denseBincount(x, weights, size, binaryOutput?)`

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **x** (tf.Tensor1D|tf.Tensor2D|TypedArray|Array) - Required - The input int tensor, rank 1 or rank 2.
- **weights** (tf.Tensor1D|tf.Tensor2D|TypedArray|Array) - Required - The weights tensor, must have the same shape as x, or a length-0 Tensor, in which case it acts as all weights equal to 1.
- **size** (number) - Required - Non-negative integer. The number of bins.
- **binaryOutput** (boolean) - Optional - Whether the kernel should count the appearance or number of occurrences. Defaults to False.

### Request Example
```json
{
  "example": "tf.denseBincount(x, weights, size, binaryOutput)"
}
```

### Response
#### Success Response (200)
- **result** (tf.Tensor1D|tf.Tensor2D) - The binned counts or weighted sums.

#### Response Example
```json
{
  "example": "tf.Tensor1D or tf.Tensor2D"
}
```
```

--------------------------------

### Create Tensor Filled with Zeros

Source: https://js.tensorflow.org/api/latest/index

Creates a tensor of a specified shape with all elements initialized to zero. The data type of the elements can also be specified. This function is supported on webgpu, webgl, wasm, and cpu.

```javascript
tf.zeros([2, 2]).print();
```

--------------------------------

### tf.buffer

Source: https://js.tensorflow.org/api/latest/index

Creates an empty tf.TensorBuffer with the specified shape and dtype. Values can be set using `buffer.set()` or `buffer.values` and then converted to a tensor using `buffer.toTensor()`.

```APIDOC
## tf.buffer

### Description
Creates an empty `tf.TensorBuffer` with the specified `shape` and `dtype`. Values are stored in CPU as `TypedArray`. Fill the buffer using `buffer.set()`, or by modifying `buffer.values` directly. When done, call `buffer.toTensor()` to obtain an immutable `tf.Tensor` with those values.

### Method
`tf.buffer(shape, dtype?, values?): tf.TensorBuffer`

### Parameters
* **shape** (number[]) - An array of integers defining the output tensor shape.
* **dtype** ('float32') - The dtype of the buffer. Defaults to 'float32'. Optional.
* **values** (DataTypeMap['float32']) - The values of the buffer as TypedArray. Defaults to zeros. Optional.

### Request Example
```javascript
// Create a buffer and set values at particular indices.
const buffer = tf.buffer([2, 2]);
buffer.set(3, 0, 0);
buffer.set(5, 1, 0);

// Convert the buffer back to a tensor.
buffer.toTensor().print();
```

### Response Example
```json
{
  "example": "tf.TensorBuffer"
}
```
```

--------------------------------

### tf.train.momentum

Source: https://js.tensorflow.org/api/latest/index

Constructs a tf.MomentumOptimizer that uses momentum gradient descent.

```APIDOC
## tf.train.momentum

### Description
Constructs a tf.MomentumOptimizer that uses momentum gradient descent.

### Method
`tf.train.momentum`

### Endpoint
N/A (This is a function call within the library)

### Parameters
#### Path Parameters
N/A

#### Query Parameters
N/A

#### Request Body
N/A

### Request Example
```javascript
const learningRate = 0.01;
const momentum = 0.9;
const optimizer = tf.train.momentum(learningRate, momentum);
```

### Response
#### Success Response (200)
Returns a `tf.MomentumOptimizer` instance.

#### Response Example
```json
{
  "optimizerType": "momentum"
}
```
```

--------------------------------

### tf.clone

Source: https://js.tensorflow.org/api/latest/index

Creates a new tensor with the same values and shape as the specified tensor.

```APIDOC
## tf.clone

### Description
Creates a new tensor that is a deep copy of the specified tensor, meaning it has the same values and shape.

### Method
`tf.clone(x): tf.Tensor`

### Parameters
* **x** (tf.Tensor|TypedArray|Array) - The tensor to clone.

### Request Example
```javascript
const x = tf.tensor([1, 2]);

x.clone().print();
```

### Response Example
```json
{
  "example": "tf.Tensor"
}
```
```

--------------------------------

### BatchNormalization Layer in TensorFlow.js

Source: https://js.tensorflow.org/api/latest/index

This snippet illustrates the use of the BatchNormalization layer in TensorFlow.js. It explains how this layer normalizes activations, maintaining mean activation close to 0 and standard deviation close to 1. Key parameters like axis, momentum, epsilon, center, and scale are demonstrated.

```javascript
const bnLayer = tf.layers.batchNormalization({
  axis: -1, // Default: features axis
  momentum: 0.99, // Default: 0.99
  epsilon: 1e-3, // Default: 1e-3
  center: true, // Default: true
  scale: true, // Default: true
  betaInitializer: 'zeros', // Default: 'zeros'
  gammaInitializer: 'ones', // Default: 'ones'
  movingMeanInitializer: 'zeros', // Default: 'zeros'
  movingVarianceInitializer: 'Ones', // Default: 'Ones'
  betaConstraint: null, // Default: null
  gammaConstraint: null, // Default: null
  betaRegularizer: null, // Default: null
  gammaRegularizer: null // Default: null
});

// Example usage within a model:
const model = tf.sequential();
model.add(tf.layers.dense({ units: 10, inputShape: [784] }));
model.add(tf.layers.batchNormalization()); // Using default parameters
model.add(tf.layers.dense({ units: 1 }));
```

--------------------------------

### Image Rotation API

Source: https://js.tensorflow.org/api/latest/index

API for rotating images with an optional offset center of rotation.

```APIDOC
## tf.image.rotateWithOffset

### Description
Rotates the input image tensor counter-clockwise with an optional offset center of rotation. Currently available in the CPU, WebGL, and WASM backends.

### Method
Not specified (assumed to be a function call)

### Endpoint
N/A

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **image** (tf.Tensor4D|TypedArray|Array) - Required - 4d tensor of shape `[batch, imageHeight, imageWidth, depth]`.
- **radians** (number) - Required - The amount of rotation.
- **fillValue** (number|[number, number, number]) - Optional - The value to fill in the empty space leftover after rotation. Can be either a single grayscale value (0-255), or an array of three numbers `[red, green, blue]` specifying the red, green, and blue channels. Defaults to `0` (black).
- **center** (number|[number, number]) - Optional - The center of rotation. Can be either a single value (0-1), or an array of two numbers `[centerX, centerY]`. Defaults to `0.5` (rotates the image around its center).

### Request Example
```javascript
tf.image.rotateWithOffset(imageTensor, Math.PI / 4, 0.5, [0.25, 0.75])
```

### Response
#### Success Response (200)
- **result** (tf.Tensor4D) - The rotated image tensor.
```

--------------------------------

### Move Models Between Storage with tf.io.moveModel in JavaScript

Source: https://js.tensorflow.org/api/latest/index

Illustrates how to move a saved model from one storage location to another using tf.io.moveModel. This function supports moving models within the same storage medium or between different ones, such as from Local Storage to IndexedDB.

```javascript
const model = tf.sequential();
model.add(tf.layers.dense({units: 1, inputShape: [10], activation: 'sigmoid'}));
await model.save('localstorage://demo/management/model1');

console.log(JSON.stringify(await tf.io.listModels()));

await tf.io.moveModel(
     'localstorage://demo/management/model1',
     'indexeddb://demo/management/model1');

console.log(JSON.stringify(await tf.io.listModels()));

await tf.io.removeModel('indexeddb://demo/management/model1');
```

--------------------------------

### mapAsync (transform)

Source: https://js.tensorflow.org/api/latest/index

Maps this dataset through an async 1-to-1 transform. The transform function should return a Promise for the transformed dataset element.

```APIDOC
## mapAsync (transform)

### Description
Maps this dataset through an async 1-to-1 transform. The transform function should return a Promise for the transformed dataset element. This transform is responsible for disposing any intermediate `Tensor`s, i.e. by wrapping its computation in `tf.tidy()`; that cannot be automated here (as it is in the synchronous `map()` case).

### Method
mapAsync

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **transform** (function) - A function mapping a dataset element to a `Promise` for a transformed dataset element.

### Request Example
```javascript
const a = tf.data.array([1, 2, 3]).mapAsync(x => new Promise(function(resolve){
  setTimeout(() => {
    resolve(x * x);
  }, Math.random()*1000 + 500);
}));
console.log(await a.toArray());
```

### Response
#### Success Response (200)
- **tf.data.Dataset** - The transformed dataset.

#### Response Example
```json
{
  "example": "[1, 4, 9]"
}
```
```

--------------------------------

### Save TensorFlow.js Model to HTTP Server

Source: https://js.tensorflow.org/api/latest/index

Demonstrates how to save a TensorFlow.js sequential model to an HTTP server using tf.io.http. This function sends model artifacts (topology and weights) as multipart/form-data. It supports custom request options like HTTP methods.

```javascript
const model = tf.sequential();
model.add(
     tf.layers.dense({units: 1, inputShape: [100], activation: 'sigmoid'})
);

const saveResult = await model.save(tf.io.http(
     'http://model-server:5000/upload', {requestInit: {method: 'PUT'}}));
console.log(saveResult);
```

```javascript
const saveResult = await model.save('http://model-server:5000/upload');
```

--------------------------------

### Gamma Distribution Sampling

Source: https://js.tensorflow.org/api/latest/index

Creates a tensor with values sampled from a gamma distribution.

```APIDOC
## POST /randomGamma

### Description
Creates a tf.Tensor with values sampled from a gamma distribution.

### Method
POST

### Endpoint
`/randomGamma`

### Parameters
#### Request Body
- **shape** (number[]) - Required - An array of integers defining the output tensor shape.
- **alpha** (number) - Required - The shape parameter of the gamma distribution.
- **beta** (number) - Optional - The inverse scale parameter of the gamma distribution. Defaults to 1.
- **dtype** ('float32'|'int32') - Optional - The data type of the output. Defaults to float32.
- **seed** (number) - Optional - The seed for the random number generator.

### Request Example
```json
{
  "shape": [2, 2],
  "alpha": 1
}
```

### Response
#### Success Response (200)
- **result** (tf.Tensor) - A tensor with values sampled from the gamma distribution.

#### Response Example
```json
{
  "result": [[0.5, 1.2], [0.8, 1.5]]
}
```
```

--------------------------------

### Model Evaluation: evaluateDataset

Source: https://js.tensorflow.org/api/latest/index

Evaluates the model using a provided dataset. This method is asynchronous and designed for evaluating with data from a tf.data.Dataset object.

```APIDOC
## evaluateDataset

### Description
Evaluate model using a dataset object. This method is asynchronous.

### Method
`async evaluateDataset(dataset, args?)

### Parameters
#### Path Parameters
* dataset (tf.data.Dataset) - Required - A dataset object. Its `iterator()` method is expected to generate a dataset iterator object, the `next()` method of which is expected to produce data batches for evaluation. The return value of the `next()` call ought to contain a boolean `done` field and a `value` field. The `value` field is expected to be an array of two tf.Tensors or an array of two nested tf.Tensor structures. Of the two items in the array, the first is the input feature(s) and the second is the output target(s).
* args (Object) - Optional - A configuration object for the dataset-based evaluation.
  * batches (number) - Optional - Number of batches to draw from the dataset object before ending the evaluation.
  * verbose (ModelLoggingVerbosity) - Optional - Verbosity mode.

### Returns
Promise<tf.Scalar|tf.Scalar[]>

### Notes
- The `steps` parameter (number of steps before declaring evaluation round finished) is ignored with the default value of `undefined`.
```

--------------------------------

### Minimize Scalar Output with TensorFlow.js Optimizer

Source: https://js.tensorflow.org/api/latest/index

Minimizes the scalar output of a function `f` by computing gradients with respect to trainable variables. Optionally returns the cost and specifies which variables to update. Requires a function returning a tf.Scalar and returns a tf.Scalar or null.

```javascript
tf.train.Optimizer.minimize(f, returnCost?, varList?)
```

--------------------------------

### Register Custom Layer Class with Package and Name in JavaScript

Source: https://js.tensorflow.org/api/latest/index

Illustrates registering a custom Layer class with both a package name ('Package') and a specified name ('MyLayer') in TensorFlow.js serialization. This allows for more organized management of custom classes.

```javascript
class MyCustomLayer extends tf.layers.Layer {
   static className = 'MyCustomLayer';

   constructor(config) {
     super(config);
   }
}
tf.serialization.registerClass(MyCustomLayer, "Package", "MyLayer");
console.log(tf.serialization.GLOBALCUSTOMOBJECT.get("Package>MyLayer"));
console.log(tf.serialization.GLOBALCUSTOMNAMES.get(MyCustomLayer));
```

--------------------------------

### Find Minimum Element-wise with tf.minimum

Source: https://js.tensorflow.org/api/latest/index

Returns the minimum of two tf.Tensors element-wise (a < b ? a : b). Supports broadcasting and can be called on a tensor or as a standalone function. Optimized for webgpu, webgl, wasm, and cpu backends.

```javascript
const a = tf.tensor1d([1, 4, 3, 16]);
const b = tf.tensor1d([1, 2, 9, 4]);

a.minimum(b).print();  // or tf.minimum(a, b)
```

```javascript
// Broadcast minimum a with b.
const a = tf.tensor1d([2, 4, 6, 8]);
const b = tf.scalar(5);

a.minimum(b).print();  // or tf.minimum(a, b)
```

--------------------------------

### tf.ceil

Source: https://js.tensorflow.org/api/latest/index

Computes the ceiling of a tensor element-wise.

```APIDOC
## tf.ceil (x)

### Description
Computes ceiling of input tf.Tensor element-wise: `ceil(x)`.

### Method
`tf.ceil` or Tensor.ceil

### Parameters
* `x` (tf.Tensor|TypedArray|Array) - The input Tensor.

### Request Example
```javascript
const x = tf.tensor1d([.6, 1.1, -3.3]);

x.ceil().print();  // or tf.ceil(x)
```

### Response Example
```javascript
// Tensor
//   [1,
//    2,
//   -3]
```
```

--------------------------------

### Generate Evenly Spaced Sequence (JavaScript)

Source: https://js.tensorflow.org/api/latest/index

Creates a 1D tensor containing an evenly spaced sequence of numbers within a specified interval. Supported backends include webgpu, webgl, wasm, and cpu.

```javascript
tf.linspace(0, 9, 10).print();
```

--------------------------------

### MaxPooling2D API

Source: https://js.tensorflow.org/api/latest/index

Provides max pooling for 2D spatial data. It reduces the spatial dimensions of the input by taking the maximum value over a window.

```APIDOC
## POST /tf.layers.maxPooling2d

### Description
Max pooling operation for spatial data. It reduces the spatial dimensions of the input by taking the maximum value over a window.

### Method
POST

### Endpoint
/tf.layers.maxPooling2d

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **args** (Object) - Required - Configuration object for the pooling layer.
  - **poolSize** (number | [number, number]) - Optional - Factors by which to downscale in each dimension [vertical, horizontal]. Defaults to the layer's default.
  - **strides** (number | [number, number]) - Optional - The size of the stride in each dimension of the pooling window. If null, defaults to `poolSize`. Defaults to the layer's default.
  - **padding** ('valid' | 'same' | 'causal') - Optional - The padding type to use for the pooling layer. Defaults to 'valid'.
  - **dataFormat** ('channelsFirst' | 'channelsLast') - Optional - The data format to use for the pooling layer. Defaults to 'channelsLast'.
  - **inputShape** (Array<number | null>) - Optional - If defined, will be used to create an input layer to insert before this layer. This argument is only applicable to input layers.
  - **batchInputShape** (Array<number | null>) - Optional - If defined, will be used to create an input layer to insert before this layer. If both `inputShape` and `batchInputShape` are defined, `batchInputShape` will be used. This argument is only applicable to input layers.
  - **batchSize** (number) - Optional - If `inputShape` is specified and `batchInputShape` is _not_ specified, `batchSize` is used to construct the `batchInputShape`.
  - **dtype** ('float32' | 'int32' | 'bool' | 'complex64' | 'string') - Optional - The data-type for this layer. Defaults to 'float32'. This argument is only applicable to input layers.
  - **name** (string) - Optional - Name for this layer.
  - **trainable** (boolean) - Optional - Whether the weights of this layer are updatable by `fit`. Defaults to true.
  - **weights** (Array<tf.Tensor>) - Optional - Initial weight values of the layer.
  - **inputDType** ('float32' | 'int32' | 'bool' | 'complex64' | 'string') - Optional - Legacy support. Do not use for new code.

### Request Example
```json
{
  "args": {
    "poolSize": [2, 2],
    "strides": [2, 2],
    "padding": "valid",
    "dataFormat": "channelsLast"
  }
}
```

### Response
#### Success Response (200)
- **MaxPooling2D** (Object) - An instance of the MaxPooling2D layer.

#### Response Example
```json
{
  "layerType": "MaxPooling2D",
  "name": "maxPooling2d_1"
}
```
```

--------------------------------

### tf.logicalAnd (a, b)

Source: https://js.tensorflow.org/api/latest/index

Returns the truth value of `a AND b` element-wise. Supports broadcasting.

```APIDOC
## POST /tf.logicalAnd

### Description
Returns the truth value of `a AND b` element-wise. Supports broadcasting.

### Method
POST

### Endpoint
/tf.logicalAnd

### Parameters
#### Request Body
- **a** (tf.Tensor|TypedArray|Array) - Required - The first input tensor. Must be of dtype bool.
- **b** (tf.Tensor|TypedArray|Array) - Required - The second input tensor. Must be of dtype bool.

### Request Example
```json
{
  "a": [false, false, true, true],
  "b": [false, true, false, true]
}
```

### Response
#### Success Response (200)
- **result** (tf.Tensor) - The resulting tensor with truth values.

#### Response Example
```json
{
  "result": [false, false, false, true]
}
```
```

--------------------------------

### tf.tensor2d

Source: https://js.tensorflow.org/api/latest/index

Creates a rank-2 tf.Tensor with the provided values, shape and dtype.

```APIDOC
## POST /api/v1/tensor2d

### Description
Creates a rank-2 tf.Tensor with the provided values, shape and dtype.
The same functionality can be achieved with tf.tensor(), but in general we recommend using tf.tensor2d() as it makes the code more readable.

### Method
POST

### Endpoint
/api/v1/tensor2d

### Parameters
#### Request Body
- **values** (TypedArray|Array) - Required - The values of the tensor. Can be nested array of numbers, or a flat array, or a TypedArray.
- **shape** (number[]) - Optional - The shape of the tensor. If not provided, it is inferred from `values`. Optional 
- **dtype** ('float32'|'int32'|'bool'|'complex64'|'string') - Optional - The data type. Optional 

### Response
#### Success Response (200)
- **tensor2d** (tf.Tensor2D) - The created rank-2 tensor.

#### Response Example
```json
{
  "tensor2d": "<Tensor2D Object>"
}
```
```

--------------------------------

### tf.mod

Source: https://js.tensorflow.org/api/latest/index

Returns the mod of a and b element-wise. `floor(x / y) * y + mod(x, y) = x`. Supports broadcasting.

```APIDOC
## tf.mod

### Description
Returns the mod of a and b element-wise. `floor(x / y) * y + mod(x, y) = x`. Supports broadcasting.

### Method
`tf.mod(a, b)` or `a.mod(b)`

### Parameters
- **a** (tf.Tensor|TypedArray|Array) - The first tensor.
- **b** (tf.Tensor|TypedArray|Array) - The second tensor. Must have the same type as `a`.

### Request Example
```javascript
const a = tf.tensor1d([1, 4, 3, 16]);
const b = tf.tensor1d([1, 2, 9, 4]);

a.mod(b).print();  // or tf.mod(a, b)
```

### Response
#### Success Response (tf.Tensor)
- **result** (tf.Tensor) - The element-wise modulo of a and b.

#### Response Example
```json
// Example output for a.mod(b):
// Tensor
//    [0, 0, 3, 0]
```
```

--------------------------------

### tf.dropout

Source: https://js.tensorflow.org/api/latest/index

Computes dropout, a technique for regularization by randomly setting some elements of the input tensor to zero.

```APIDOC
## POST /tf.dropout

### Description
Computes dropout, a technique for regularization by randomly setting some elements of the input tensor to zero.

### Method
POST

### Endpoint
/tf.dropout

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **x** (tf.Tensor|TypedArray|Array) - Required - A floating point Tensor or TensorLike.
- **rate** (number) - Required - A float in the range [0, 1). The probability that each element of x is discarded.
- **noiseShape** (number[]) - Optional - An array of numbers of type int32, representing the shape for randomly generated keep/drop flags. If the noiseShape has null value, it will be automatically replaced with the x's relative dimension size.
- **seed** (number|string) - Optional - Used to create random seeds.

### Request Example
```json
{
  "x": [1, 2, 2, 1],
  "rate": 0.75
}
```

### Response
#### Success Response (200)
- **tensor** (tf.Tensor) - The resulting tensor after dropout is applied.

#### Response Example
```json
{
  "tensor": "[Tensor object representation]"
}
```
```

--------------------------------

### Load and Process CSV Data with TensorFlow.js

Source: https://js.tensorflow.org/api/latest/index

The tf.data.csv() function creates a CSVDataset from a URL or local path. It can parse CSV files, identify label columns, and prepare data for model training through operations like mapping and batching.

```javascript
const csvUrl =
'https://storage.googleapis.com/tfjs-examples/multivariate-linear-regression/data/boston-housing-train.csv';

async function run() {
   // We want to predict the column "medv", which represents a median value of
   // a home (in $1000s), so we mark it as a label.
   const csvDataset = tf.data.csv(
     csvUrl, {
       columnConfigs: {
         medv: {
           isLabel: true
         }
       }
     });

   // Number of features is the number of column names minus one for the label
   // column.
   const numOfFeatures = (await csvDataset.columnNames()).length - 1;

   // Prepare the Dataset for training.
   const flattenedDataset =
     csvDataset
     .map(({xs, ys}) =>
       {
         // Convert xs(features) and ys(labels) from object form (keyed by
         // column name) to array form.
         return {xs:Object.values(xs), ys:Object.values(ys)};
       })
     .batch(10);

   // Define the model.
   const model = tf.sequential();
   model.add(tf.layers.dense({
     inputShape: [numOfFeatures],
     units: 1
   }));
   model.compile({
     optimizer: tf.train.sgd(0.000001),
     loss: 'meanSquaredError'
   });

   // Fit the model using the prepared Dataset
   return model.fitDataset(flattenedDataset, {
     epochs: 10,
     callbacks: {
       onEpochEnd: async (epoch, logs) => {
         console.log(epoch + ':' + logs.loss);
       }
     }
   });
}

await run();
```

--------------------------------

### Apply SimpleRNNCell in TensorFlow.js

Source: https://js.tensorflow.org/api/latest/index

Demonstrates how to apply a SimpleRNNCell in TensorFlow.js. The cell processes input data for a single time step and returns the cell's output for that step. The output shape includes the batch size and the number of units in the cell.

```javascript
const cell = tf.layers.simpleRNNCell({units: 2});
const input = tf.input({shape: [10]});
const output = cell.apply(input);

console.log(JSON.stringify(output.shape));
// [null, 10]: This is the cell's output at a single time step. The 1st
// dimension is the unknown batch size.
```

--------------------------------

### LSTM Layer Usage in TensorFlow.js

Source: https://js.tensorflow.org/api/latest/index

Demonstrates how to create and apply an LSTM layer in TensorFlow.js. It shows how to define input shapes and interpret output shapes based on layer configurations like `returnSequences`. This layer is suitable for processing sequential data.

```javascript
const lstm = tf.layers.lstm({units: 8, returnSequences: true});

// Create an input with 10 time steps.
const input = tf.input({shape: [10, 20]});
const output = lstm.apply(input);

console.log(JSON.stringify(output.shape));
// [null, 10, 8]: 1st dimension is unknown batch size; 2nd dimension is the
// same as the sequence length of `input`, due to `returnSequences`: `true`;
// 3rd dimension is the `LSTMCell`'s number of units.
```

--------------------------------

### tf.signal.stft

Source: https://js.tensorflow.org/api/latest/index

Computes the Short-time Fourier Transform of signals.

```APIDOC
## POST /tf.signal.stft

### Description
Computes the Short-time Fourier Transform of signals.
See: https://en.wikipedia.org/wiki/Short-time_Fourier_transform

### Method
POST

### Endpoint
/tf.signal.stft

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **signal** (tf.Tensor1D) - Required - 1-dimensional real value tensor.
- **frameLength** (number) - Required - The window length of samples.
- **frameStep** (number) - Required - The number of samples to step.
- **fftLength** (number) - Optional - The size of the FFT to apply.
- **windowFn** ((length: number) => tf.Tensor1D) - Optional - A callable that takes a window length and returns 1-d tensor.

### Request Example
```json
{
  "signal": [1, 1, 1, 1, 1],
  "frameLength": 3,
  "frameStep": 1
}
```

### Response
#### Success Response (200)
- **tensor** (tf.Tensor) - The STFT tensor.

#### Response Example
```json
{
  "tensor": "[Tensor object representation]"
}
```
```

--------------------------------

### PReLU Activation Layer

Source: https://js.tensorflow.org/api/latest/index

The PReLU (Parameterized Rectified Linear Unit) layer is a parameterized version of LeakyReLU. It follows `f(x) = alpha * x for x < 0.` and `f(x) = x for x >= 0.` wherein `alpha` is a trainable weight.

```APIDOC
## PReLU Activation Layer

### Description
The Parameterized version of a leaky rectified linear unit. It follows `f(x) = alpha * x for x < 0.` `f(x) = x for x >= 0.` wherein `alpha` is a trainable weight.

### Method
`tf.layers.prelu`

### Endpoint
N/A (Layer definition)

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **args** (Object) - Optional
- **alphaInitializer** (tf.initializers.Initializer|'constant'|'glorotNormal'|'glorotUniform'|'heNormal'|'heUniform'|'identity'| 'leCunNormal'|'leCunUniform'|'ones'|'orthogonal'|'randomNormal'| 'randomUniform'|'truncatedNormal'|'varianceScaling'|'zeros'|string) - Optional - Initializer for the learnable alpha.
- **alphaRegularizer** (Regularizer) - Optional - Regularizer for the learnable alpha.
- **alphaConstraint** (tf.constraints.Constraint) - Optional - Constraint for the learnable alpha.
- **sharedAxes** (number|number[]) - Optional - The axes along which to share learnable parameters for the activation function. For example, if the incoming feature maps are from a 2D convolution with output shape `[numExamples, height, width, channels]`, and you wish to share parameters across space (height and width) so that each filter channels has only one set of parameters, set `shared_axes: [1, 2]`.
- **inputShape** ((null | number)[]) - Optional - If defined, will be used to create an input layer to insert before this layer. If both `inputShape` and `batchInputShape` are defined, `batchInputShape` will be used. This argument is only applicable to input layers (the first layer of a model).
- **batchInputShape** ((null | number)[]) - Optional - If defined, will be used to create an input layer to insert before this layer. If both `inputShape` and `batchInputShape` are defined, `batchInputShape` will be used. This argument is only applicable to input layers (the first layer of a model).
- **batchSize** (number) - Optional - If `inputShape` is specified and `batchInputShape` is _not_ specified, `batchSize` is used to construct the `batchInputShape`: `[batchSize, ...inputShape]`
- **dtype** ('float32'|'int32'|'bool'|'complex64'|'string') - Optional - The data-type for this layer. Defaults to 'float32'. This argument is only applicable to input layers (the first layer of a model).
- **name** (string) - Optional - Name for this layer.
- **trainable** (boolean) - Optional - Whether the weights of this layer are updatable by `fit`. Defaults to true.
- **weights** (tf.Tensor[]) - Optional - Initial weight values of the layer.
- **inputDType** ('float32'|'int32'|'bool'|'complex64'|'string') - Optional - Legacy support. Do not use for new code.

### Request Example
```json
{
  "sharedAxes": [1, 2],
  "alphaInitializer": "ones"
}
```

### Response
#### Success Response (200)
- **Layer Output** (Tensor) - Same shape as the input.

#### Response Example
```json
// Tensor output representing the layer's activation
```
```

--------------------------------

### tf.initializers.truncatedNormal

Source: https://js.tensorflow.org/api/latest/index

Initializer that generates random values from a truncated normal distribution, discarding values more than two standard deviations from the mean.

```APIDOC
## tf.initializers.truncatedNormal (args)

### Description
Initializer that generates random values initialized to a truncated normal distribution. These values are similar to values from a `RandomNormal` except that values more than two standard deviations from the mean are discarded and re-drawn. This is the recommended initializer for neural network weights and filters.

### Method
Function

### Endpoint
N/A (Client-side JavaScript function)

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
* **args** (Object) - Configuration object for the initializer.
* **mean** (number) - Optional. Mean of the random values to generate. Defaults to 0.
* **stddev** (number) - Optional. Standard deviation of the random values to generate. Defaults to 1.
* **seed** (number) - Optional. Used to seed the random generator.

### Request Example
```json
{
  "args": {
    "mean": 0.1,
    "stddev": 0.3,
    "seed": 106
  }
}
```

### Response
#### Success Response (tf.initializers.Initializer)
Returns an instance of tf.initializers.Initializer.

#### Response Example
```json
{
  "instance": "tf.initializers.Initializer"
}
```
```

--------------------------------

### Construct tf.MomentumOptimizer in JavaScript

Source: https://js.tensorflow.org/api/latest/index

Constructs a tf.MomentumOptimizer that uses momentum gradient descent. This optimizer helps to accelerate gradient descent in the relevant direction and dampens oscillations.

```javascript
const learningRate = 0.01;
const momentum = 0.9;
const optimizer = tf.train.momentum(learningRate, momentum);

```

--------------------------------

### tf.zerosLike

Source: https://js.tensorflow.org/api/latest/index

Creates a tf.Tensor with all elements set to 0 with the same shape as the given tensor.

```APIDOC
## POST /tf.zerosLike

### Description
Creates a tf.Tensor with all elements set to 0 with the same shape as the given tensor.

### Method
POST

### Endpoint
/tf.zerosLike

### Parameters
#### Query Parameters
- **x** (tf.Tensor|TypedArray|Array) - Required - The tensor of required shape.

### Request Example
```json
{
  "x": [1, 2]
}
```

### Response
#### Success Response (200)
- **tf.Tensor** - A tensor with the same shape as the input, filled with zeros.

#### Response Example
```json
{
  "tensor": [0, 0]
}
```
```

--------------------------------

### tf.maxPoolWithArgmax

Source: https://js.tensorflow.org/api/latest/index

Computes the 2D max pooling of an image with Argmax index. The indices in argmax are flattened.

```APIDOC
## tf.maxPoolWithArgmax

### Description
Computes the 2D max pooling of an image with Argmax index. The indices in argmax are flattened, so that a maximum value at position `[b, y, x, c]` becomes flattened index: `(y * width + x) * channels + c` if include_batch_in_index is False; `((b * height + y) * width + x) * channels +c` if include_batch_in_index is True.
The indices returned are always in `[0, height) x [0, width)` before flattening.

### Method
`tf.maxPoolWithArgmax`

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
* **x** (tf.Tensor4D|TypedArray|Array) - The input tensor, of rank 4 or rank 3 of shape `[batch, height, width, inChannels]`. If rank 3, batch of 1 is assumed.
* **filterSize** ([number, number]|number) - The filter size: `[filterHeight, filterWidth]`. If `filterSize` is a single number, then `filterHeight == filterWidth`.
* **strides** ([number, number]|number) - The strides of the pooling: `[strideHeight, strideWidth]`. If `strides` is a single number, then `strideHeight == strideWidth`.
* **pad** ('valid'|'same'|number) - The type of padding algorithm. `same` and stride 1: output will be of same size as input, regardless of filter size. `valid`: output will be smaller than input if filter is larger than 1x1. For more info, see this guide: https://www.tensorflow.org/api_docs/python/tf/nn/convolution
* **includeBatchInIndex** (boolean) - Optional

### Request Example
```json
{
  "x": "input tensor",
  "filterSize": [2, 2],
  "strides": [1, 1],
  "pad": "valid",
  "includeBatchInIndex": false
}
```

### Response
#### Success Response (200)
* **output** (tf.Tensor4D|tf.Tensor5D) - The output tensor.
* **argmax** (tf.Tensor4D|tf.Tensor5D) - The argmax tensor.

#### Response Example
```json
{
  "output": "output tensor",
  "argmax": "argmax tensor"
}
```
```

--------------------------------

### Retrieve Tensor Data Synchronously (JavaScript)

Source: https://js.tensorflow.org/api/latest/index

The `dataSync()` method retrieves tensor data synchronously, returning the data as a typed array. This method blocks the UI thread and should be used cautiously to avoid performance degradation.

```javascript
const data = await tensor.dataSync();
```

--------------------------------

### tf.where (condition, a, b)

Source: https://js.tensorflow.org/api/latest/index

Returns the elements, either `a` or `b` depending on the `condition`. If the condition is true, select from `a`, otherwise select from `b`.

```APIDOC
## POST /tf.where

### Description
Returns the elements, either `a` or `b` depending on the `condition`. If the condition is true, select from `a`, otherwise select from `b`.

### Method
POST

### Endpoint
/tf.where

### Parameters
#### Request Body
- **condition** (tf.Tensor|TypedArray|Array) - Required - The input condition. Must be of dtype bool.
- **a** (tf.Tensor|TypedArray|Array) - Required - If `condition` is rank 1, `a` may have a higher rank but its first dimension must match the size of `condition`.
- **b** (tf.Tensor|TypedArray|Array) - Required - A tensor with the same dtype as `a` and with shape that is compatible with `a`.

### Request Example
```json
{
  "condition": [false, false, true],
  "a": [1, 2, 3],
  "b": [-1, -2, -3]
}
```

### Response
#### Success Response (200)
- **result** (tf.Tensor) - The resulting tensor with selected elements.

#### Response Example
```json
{
  "result": [-1, -2, 3]
}
```
```

--------------------------------

### Create Complex Numbers from Real and Imaginary Parts (JavaScript)

Source: https://js.tensorflow.org/api/latest/index

Combines real and imaginary parts into complex numbers. Requires input tensors to have the same shape. Supported backends include webgpu, webgl, wasm, and cpu.

```javascript
const real = tf.tensor1d([2.25, 3.25]);
const imag = tf.tensor1d([4.75, 5.75]);
const complex = tf.complex(real, imag);

complex.print();
```

--------------------------------

### TensorFlow.js: Matrix Transpose using einsum

Source: https://js.tensorflow.org/api/latest/index

Performs a matrix transpose operation on a 2D tensor using `einsum`. The equation 'ij->ji' swaps the row and column indices.

```javascript
const x = tf.tensor2d([[1, 2], [3, 4]]);
x.print();
tf.einsum('ij->ji', x).print();
```

--------------------------------

### tf.confusionMatrix: Compute Confusion Matrix

Source: https://js.tensorflow.org/api/latest/index

Computes the confusion matrix from true labels and predicted labels. The function takes true labels, predicted labels, and the total number of classes as input. Supports multiple backends like webgpu, webgl, wasm, and cpu.

```javascript
const labels = tf.tensor1d([0, 1, 2, 1, 0], 'int32');
const predictions = tf.tensor1d([0, 2, 2, 1, 0], 'int32');
const numClasses = 3;
const out = tf.math.confusionMatrix(labels, predictions, numClasses);
out.print();
// Expected output matrix:
// [[2, 0, 0],
//  [0, 1, 1],
//  [0, 0, 1]]
```

--------------------------------

### tf.sparseToDense

Source: https://js.tensorflow.org/api/latest/index

Converts a sparse representation into a dense tensor. It builds an array dense with a specified shape, placing values at specified indices.

```APIDOC
## POST /tf.sparseToDense

### Description
Converts a sparse representation into a dense tensor. It builds an array dense with a specified shape, placing values at specified indices. If indices are repeated, the final value is summed over all values for those indices.

### Method
POST

### Endpoint
/tf.sparseToDense

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **sparseIndices** (tf.Tensor|TypedArray|Array) - Required - A 0-D, 1-D, or 2-D Tensor of type int32. `sparseIndices[i]` contains the complete index where `sparseValues[i]` will be placed.
- **sparseValues** (tf.Tensor|TypedArray|Array) - Required - A 0-D or 1-D Tensor. Values corresponding to each row of `sparseIndices`, or a scalar value to be used for all sparse indices.
- **outputShape** (number[]) - Required - Shape of the dense output tensor. The type is inferred.
- **defaultValue** (tf.Scalar|ScalarLike) - Optional - Scalar. Value to set for indices not specified in `sparseIndices`. Defaults to zero.

### Request Example
```json
{
  "sparseIndices": [4, 5, 6, 1, 2, 3],
  "sparseValues": [10, 11, 12, 13, 14, 15],
  "outputShape": [8],
  "defaultValue": 0
}
```

### Response
#### Success Response (200)
- **output** (tf.Tensor) - The resulting dense tensor.

#### Response Example
```json
{
  "output": "[[...]]"
}
```
```

--------------------------------

### Define a Dense Layer in TensorFlow.js

Source: https://js.tensorflow.org/api/latest/index

Defines a dense (fully connected) layer in TensorFlow.js. This layer performs the operation: `output = activation(dot(input, kernel) + bias)`. It accepts various parameters to configure the units, activation function, bias, initializers, constraints, regularizers, and input shape.

```javascript
tf.layers.dense({
  units: 10,
  activation: 'relu',
  kernelInitializer: 'glorotUniform',
  biasInitializer: 'zeros',
  inputShape: [5]
});
```

--------------------------------

### Perform Batch Normalization

Source: https://js.tensorflow.org/api/latest/index

Applies batch normalization to the input tensor. This function requires mean, variance, and optionally offset and scale tensors. It can handle inputs where mean, variance, offset, and scale have the same shape as the input or are 1D tensors representing the depth dimension.

```javascript
const x = tf.tensor2d([1, 2, 3, 4], [2, 2]);
const mean = tf.tensor1d([0, 0]);
const variance = tf.tensor1d([1, 1]);

tf.batchNorm(x, mean, variance).print();
```

--------------------------------

### Generate Predictions with tf.Model.predict

Source: https://js.tensorflow.org/api/latest/index

Generates output predictions for input samples using a TensorFlow.js model. Predictions are computed in batches. The 'step' mode is not supported. It takes input data and optional arguments like batch size and verbosity.

```javascript
const model = tf.sequential({
   layers: [tf.layers.dense({units: 1, inputShape: [10]})]
});
model.predict(tf.ones([8, 10]), {batchSize: 4}).print();
```

--------------------------------

### Model Prediction: predictOnBatch

Source: https://js.tensorflow.org/api/latest/index

Returns predictions for a single batch of samples. This method is used for making predictions on a specific batch of data.

```APIDOC
## predictOnBatch

### Description
Returns predictions for a single batch of samples.

### Method
`predictOnBatch(x)

### Parameters
#### Path Parameters
* x (tf.Tensor|tf.Tensor[]) - Required - Input samples, as a Tensor (for models with exactly one input) or an array of Tensors (for models with more than one input).

### Request Example
```javascript
const model = tf.sequential({
   layers: [tf.layers.dense({units: 1, inputShape: [10]})]
});
model.predictOnBatch(tf.ones([8, 10])).print();
```

### Returns
tf.Tensor|tf.Tensor[]

### Notes
- Supported backends: webgpu, webgl, wasm, cpu.
```

--------------------------------

### tf.LayersModel.evaluate

Source: https://js.tensorflow.org/api/latest/index

Evaluates the model's performance on test data, returning the loss value and metrics.

```APIDOC
## POST /websites/js_tensorflow_api/layersModel/evaluate

### Description
Returns the loss value & metrics values for the model in test mode. Loss and metrics are specified during `compile()`, which needs to happen before calls to `evaluate()`. Computation is done in batches.

### Method
POST

### Endpoint
/websites/js_tensorflow_api/layersModel/evaluate

### Parameters
#### Path Parameters
- **x** (tf.Tensor|tf.Tensor[]) - Required - tf.Tensor of test data, or an `Array` of tf.Tensors if the model has multiple inputs.
- **y** (tf.Tensor|tf.Tensor[]) - Required - tf.Tensor of target data, or an `Array` of tf.Tensors if the model has multiple outputs.

#### Request Body
- **args** (Object) - Optional - A `ModelEvaluateArgs` object, containing optional fields.
  - **batchSize** (number) - Optional - Batch size (Integer). Defaults to 32.
  - **verbose** (ModelLoggingVerbosity) - Optional - Verbosity mode.
  - **sampleWeight** (tf.Tensor) - Optional - Tensor of weights to weight the contribution of different samples to the loss and metrics.

### Request Example
```javascript
const model = tf.sequential({
   layers: [tf.layers.dense({units: 1, inputShape: [10]})]
});
model.compile({optimizer: 'sgd', loss: 'meanSquaredError'});
const result = model.evaluate(
     tf.ones([8, 10]), tf.ones([8, 1]), {batchSize: 4});
result.print();
```

### Response
#### Success Response (200)
- **tf.Scalar|tf.Tensor** - The loss value & metrics values for the model in test mode.
```

--------------------------------

### Configuring Stateful RNN Layers

Source: https://js.tensorflow.org/api/latest/index

To enable statefulness in RNN layers, you must set `stateful: true`, specify a fixed `batchInputShape` or `batchShape`, and set `shuffle: false` during model fitting. This ensures states are reused across batches.

```typescript
// For sequential models:
model.add(tf.layers.gru({..., stateful: true, batchInputShape: [32, 10, 100]}));

// For functional models:
const input = tf.input({batchShape: [32, 10, 100]});
const gruLayer = tf.layers.gru({..., stateful: true})(input);

// When fitting the model:
model.fit(xTrain, yTrain, {shuffle: false});
```

--------------------------------

### tf.asinh

Source: https://js.tensorflow.org/api/latest/index

Computes the inverse hyperbolic sine of a tensor element-wise.

```APIDOC
## tf.asinh (x)

### Description
Computes inverse hyperbolic sin of the input tf.Tensor element-wise: `asinh(x)`.

### Method
`tf.asinh` or Tensor.asinh

### Parameters
* `x` (tf.Tensor|TypedArray|Array) - The input tensor.

### Request Example
```javascript
const x = tf.tensor1d([0, 1, -1, .7]);

x.asinh().print();  // or tf.asinh(x)
```

### Response Example
```javascript
// Tensor
//   [0,
//    0.8813736,
//   -0.8813736,
//    0.6580637]
```
```

--------------------------------

### tf.onesLike

Source: https://js.tensorflow.org/api/latest/index

Creates a tensor with the same shape as the input tensor, filled with 1s.

```APIDOC
## POST /websites/js_tensorflow_api/tf.onesLike

### Description
Creates a tf.Tensor with all elements set to 1 with the same shape as the given tensor.

### Method
POST

### Endpoint
/websites/js_tensorflow_api/tf.onesLike

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **x** (tf.Tensor) - The input tensor whose shape will be used for the output.

### Request Example
```json
{
  "x": [[1, 2], [3, 4]]
}
```

### Response
#### Success Response (200)
- **ones_like_tensor** (tf.Tensor) - A tensor with the same shape as `x` and all elements set to 1.

#### Response Example
```json
{
  "ones_like_tensor": [[1, 1], [1, 1]]
}
```
```

--------------------------------

### Draw Tensor to Canvas (Browser JavaScript)

Source: https://js.tensorflow.org/api/latest/index

Draws a `tf.Tensor` onto an HTML canvas. Handles float32 tensors (assumed range [0-1]) and int32 tensors (assumed range [0-255]). Supports different tensor shapes for grayscale, RGB, and RGBA rendering. Allows customization of image and canvas context options.

```javascript
tf.browser.draw (image, canvas, options?) function Source
Draws a tf.Tensor to a canvas.
When the dtype of the input is 'float32', we assume values in the range [0-1]. Otherwise, when input is 'int32', we assume values in the range [0-255].
Parameters:
  * image (tf.Tensor2D|tf.Tensor3D|TypedArray|Array) The tensor to draw on the canvas. Must match one of these shapes:
    * Rank-2 with shape `[height, width`]: Drawn as grayscale.
    * Rank-3 with shape `[height, width, 1]`: Drawn as grayscale.
    * Rank-3 with shape `[height, width, 3]`: Drawn as RGB with alpha set in `imageOptions` (defaults to 1, which is opaque).
    * Rank-3 with shape `[height, width, 4]`: Drawn as RGBA.
  * canvas (HTMLCanvasElement) The canvas to draw to.
  * options (Object) The configuration arguments for image to be drawn and the canvas to draw to. Optional 
  * imageOptions (ImageOptions) Optional. An object of options to customize the values of image tensor.
  * contextOptions (ContextOptions) Optional. An object to configure the context of the canvas to draw to.
Returns: void
```

--------------------------------

### Set Value in tf.TensorBuffer (JavaScript)

Source: https://js.tensorflow.org/api/latest/index

The `set()` method of a tf.TensorBuffer allows you to set a value at a specific location within the buffer. This is useful for incrementally building a tensor before converting it.

```javascript
buffer.set(value, loc1, loc2, ...);
```

--------------------------------

### Apply Softmax Activation Layer in TensorFlow.js

Source: https://js.tensorflow.org/api/latest/index

Implements a Softmax activation layer in TensorFlow.js. This layer normalizes outputs across a specified axis, typically the last one, ensuring they sum to 1. It takes optional arguments to configure the axis and input properties.

```javascript
tf.layers.softmax({
  axis: -1, // Default axis for softmax normalization
  inputShape: [10] // Example input shape if this is the first layer
});
```

--------------------------------

### tf.notEqual (a, b)

Source: https://js.tensorflow.org/api/latest/index

Returns the truth value of (a != b) element-wise. Supports broadcasting.

```APIDOC
## POST /tf.notEqual

### Description
Returns the truth value of (a != b) element-wise. Supports broadcasting.

### Method
POST

### Endpoint
/tf.notEqual

### Parameters
#### Request Body
- **a** (tf.Tensor|TypedArray|Array) - Required - The first input tensor.
- **b** (tf.Tensor|TypedArray|Array) - Required - The second input tensor. Must have the same dtype as `a`.

### Request Example
```json
{
  "a": [1, 2, 3],
  "b": [0, 2, 3]
}
```

### Response
#### Success Response (200)
- **result** (tf.Tensor) - The resulting tensor with truth values.

#### Response Example
```json
{
  "result": [true, false, false]
}
```
```

--------------------------------

### tf.tensor3d

Source: https://js.tensorflow.org/api/latest/index

Creates a rank-3 tf.Tensor with the provided values, shape and dtype. Recommended for readability over tf.tensor().

```APIDOC
## tf.tensor3d

### Description
Creates a rank-3 tf.Tensor with the provided values, shape and dtype. This function is recommended for code readability compared to using the generic `tf.tensor()`.

### Method
`tf.tensor3d(values, shape?, dtype?): tf.Tensor3D`

### Parameters
* **values** (TypedArray|Array) - The values of the tensor. Can be a nested array of numbers, a flat array, or a TypedArray.
* **shape** ([number, number, number]) - The shape of the tensor. If not provided, it is inferred from `values`. Optional.
* **dtype** ('float32'|'int32'|'bool'|'complex64'|'string') - The data type of the tensor. Optional.

### Request Example
```javascript
// Pass a nested array.
tf.tensor3d([[[1], [2]], [[3], [4]]]).print();

// Pass a flat array and specify a shape.
tf.tensor3d([1, 2, 3, 4], [2, 2, 1]).print();
```

### Response Example
```json
{
  "example": "tf.Tensor3D"
}
```
```

--------------------------------

### 3D Average Pooling - JavaScript

Source: https://js.tensorflow.org/api/latest/index

Computes the 3D average pooling of an input tensor. Supports WebGPU, WebGL, Wasm, and CPU backends. It requires the input tensor, filter size, strides, padding type, and an optional dimension rounding mode and data format.

```javascript
const x = tf.tensor5d([1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 2, 2, 1]);
const result = tf.avgPool3d(x, 2, 1, 'valid');
result.print();
```

--------------------------------

### tf.tensor3d

Source: https://js.tensorflow.org/api/latest/index

Creates a rank-3 tf.Tensor with the provided values, shape and dtype.

```APIDOC
## POST /api/v1/tensor3d

### Description
Creates a rank-3 tf.Tensor with the provided values, shape and dtype.

### Method
POST

### Endpoint
/api/v1/tensor3d

### Parameters
#### Request Body
- **values** (TypedArray|Array) - Required - The values of the tensor. Can be nested array of numbers, or a flat array, or a TypedArray.
- **shape** (number[]) - Optional - The shape of the tensor. If not provided, it is inferred from `values`. Optional 
- **dtype** ('float32'|'int32'|'bool'|'complex64'|'string') - Optional - The data type. Optional 

### Response
#### Success Response (200)
- **tensor3d** (tf.Tensor3D) - The created rank-3 tensor.

#### Response Example
```json
{
  "tensor3d": "<Tensor3D Object>"
}
```
```

--------------------------------

### Create Diagonal Tensor from Matrix (JavaScript)

Source: https://js.tensorflow.org/api/latest/index

Constructs a diagonal tensor from a 2D tensor. The output tensor's rank is twice the input tensor's rank, padded with zeros. Supported backends include webgpu, webgl, wasm, and cpu.

```javascript
const x = tf.tensor2d([1, 2, 3, 4, 5, 6, 7, 8], [4, 2]);

tf.diag(x).print();
```

--------------------------------

### tf.pow

Source: https://js.tensorflow.org/api/latest/index

Computes the power of one tf.Tensor to another. Supports broadcasting.

```APIDOC
## tf.pow

### Description
Computes the power of one tf.Tensor to another. Supports broadcasting.

### Method
`tf.pow(base, exp)` or `base.pow(exp)`

### Parameters
- **base** (tf.Tensor|TypedArray|Array) - The base tf.Tensor to pow element-wise.
- **exp** (tf.Tensor|TypedArray|Array) - The exponent tf.Tensor to pow element-wise.

### Request Example
```javascript
const a = tf.tensor([[2, 3], [4, 5]])
const b = tf.tensor([[1, 2], [3, 0]]).toInt();

a.pow(b).print();  // or tf.pow(a, b)
```

### Response
#### Success Response (tf.Tensor)
- **result** (tf.Tensor) - The element-wise result of base raised to the power of exp.

#### Response Example
```json
// Example output for a.pow(b):
// Tensor
//    [[2, 9],
//     [64, 1]]
```
```

--------------------------------

### Check for Infinity Elements with tf.isInf

Source: https://js.tensorflow.org/api/latest/index

Returns a boolean tensor indicating which elements of the input tensor are Infinity or -Infinity. It takes a tensor as input and returns a boolean tensor. Available for webgpu, webgl, wasm, and cpu backends.

```javascript
const x = tf.tensor1d([NaN, Infinity, -Infinity, 0, 1]);

x.isInf().print();  // or tf.isNaN(x)
```

--------------------------------

### Create Generic Model with tf.model() in JavaScript

Source: https://js.tensorflow.org/api/latest/index

Defines a generic model by connecting layers. It supports arbitrary, acyclic graphs of layers, unlike the more restrictive tf.sequential(). Inputs and outputs must be explicitly specified. This function is useful for complex model architectures.

```javascript
const input = tf.input({shape: [5]});

const denseLayer1 = tf.layers.dense({units: 10, activation: 'relu'});
const denseLayer2 = tf.layers.dense({units: 4, activation: 'softmax'});

const output = denseLayer2.apply(denseLayer1.apply(input));

const model = tf.model({inputs: input, outputs: output});

model.predict(tf.ones([2, 5])).print();

```

--------------------------------

### Calculate Modulo Element-wise with tf.mod

Source: https://js.tensorflow.org/api/latest/index

Returns the modulo of two tf.Tensors element-wise, satisfying the equation floor(x / y) * y + mod(x, y) = x. Supports broadcasting and can be called on a tensor or as a standalone function. Optimized for webgpu, webgl, wasm, and cpu backends.

```javascript
const a = tf.tensor1d([1, 4, 3, 16]);
const b = tf.tensor1d([1, 2, 9, 4]);

a.mod(b).print();  // or tf.mod(a, b)
```

```javascript
// Broadcast a mod b.
const a = tf.tensor1d([2, 4, 6, 8]);
const b = tf.scalar(5);

a.mod(b).print();  // or tf.mod(a, b)
```

--------------------------------

### Compute 1D Inverse Discrete Fourier Transform - tf.spectral.ifft

Source: https://js.tensorflow.org/api/latest/index

Computes the inverse 1-dimensional discrete Fourier transform over the inner-most dimension of a complex input tensor. This operation supports multiple backends including WebGPU, WebGL, WASM, and CPU.

```javascript
const real = tf.tensor1d([1, 2, 3]);
const imag = tf.tensor1d([1, 2, 3]);
const x = tf.complex(real, imag);

x.ifft().print();  // tf.spectral.ifft(x).print();

```

--------------------------------

### Convert Sparse to Dense Tensor with TensorFlow.js

Source: https://js.tensorflow.org/api/latest/index

Converts a sparse representation into a dense tensor. It builds a dense tensor of a specified shape, filling in values based on sparse indices and values. Supports optional default values. Accepts sparse indices, sparse values, output shape, and an optional default value.

```javascript
const indices = tf.tensor1d([4, 5, 6, 1, 2, 3], 'int32');
const values = tf.tensor1d([10, 11, 12, 13, 14, 15], 'float32');
const shape = [8];
tf.sparseToDense(indices, values, shape).print();
```

--------------------------------

### Create Tensor Filled with Ones (JavaScript)

Source: https://js.tensorflow.org/api/latest/index

Creates a tensor of a specified shape with all elements initialized to 1. The dtype can be specified, defaulting to 'float32'. Supported backends include webgpu, webgl, wasm, and cpu.

```javascript
tf.ones([2, 2]).print();
```

--------------------------------

### Create TensorFlow.js Dataset from Array

Source: https://js.tensorflow.org/api/latest/index

The tf.data.array() function creates a Dataset from an array of elements. It supports arrays of objects, numbers, strings, or TensorFlow.js Tensors, preparing data for machine learning tasks.

```javascript
const a = tf.data.array([{'item': 1}, {'item': 2}, {'item': 3}]);
await a.forEachAsync(e => console.log(e));
```

```javascript
const a = tf.data.array([4, 5, 6]);
await a.forEachAsync(e => console.log(e));
```

--------------------------------

### Create a 1D Convolutional Layer with TensorFlow.js

Source: https://js.tensorflow.org/api/latest/index

This snippet demonstrates how to create a 1D convolutional layer using tf.layers.conv1d. It specifies key parameters like filters, kernel size, and activation function. This layer is useful for processing sequential data.

```javascript
const convLayer = tf.layers.conv1d({
  filters: 32,
  kernelSize: 3,
  activation: 'relu',
  inputShape: [10, 128] // Example input shape for sequences of 10 vectors of 128 dimensions
});
console.log(convLayer);
```

--------------------------------

### Create Tensor from Pixels (Browser JavaScript)

Source: https://js.tensorflow.org/api/latest/index

Creates a `tf.Tensor` from various pixel data sources like `ImageData`, `HTMLImageElement`, `HTMLCanvasElement`, etc. Supports specifying the number of channels, defaulting to 3 (ignoring the alpha channel). The provided code snippet demonstrates creating a tensor from `ImageData`.

```javascript
tf.browser.fromPixels (pixels, numChannels?) function Source
Creates a tf.Tensor from an image.
```
const image = new ImageData(1, 1);
image.data[0] = 100;
image.data[1] = 150;
image.data[2] = 200;
image.data[3] = 255;

tf.browser.fromPixels(image).print();

```

EditRun
  * webgpu
  * webgl
  * wasm
  * cpu

Parameters:
  * pixels (PixelData|ImageData|HTMLImageElement|HTMLCanvasElement| HTMLVideoElement|ImageBitmap) The input image to construct the tensor from. The supported image types are all 4-channel. You can also pass in an image object with following attributes: `{data: Uint8Array; width: number; height: number}`
  * numChannels (number) The number of channels of the output tensor. A numChannels value less than 4 allows you to ignore channels. Defaults to 3 (ignores alpha channel of input image). Optional 

Returns: tf.Tensor3D
```

--------------------------------

### repeat (count?)

Source: https://js.tensorflow.org/api/latest/index

Repeats this dataset `count` times. If count is undefined or negative, the dataset is repeated indefinitely.

```APIDOC
## repeat (count?)

### Description
Repeats this dataset `count` times. Note: If this dataset is a function of global state (e.g. a random number generator), then different repetitions may produce different elements.

### Method
repeat

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **count** (number) - (Optional) An integer, representing the number of times the dataset should be repeated. The default behavior (if `count` is `undefined` or negative) is for the dataset be repeated indefinitely.

### Request Example
```javascript
const a = tf.data.array([1, 2, 3]).repeat(3);
a.forEachAsync(e => console.log(e));
```

### Response
#### Success Response (200)
- **tf.data.Dataset** - The repeated dataset.

#### Response Example
```json
{
  "example": "Dataset object"
}
```
```

--------------------------------

### tf.image.grayscaleToRGB

Source: https://js.tensorflow.org/api/latest/index

Converts a grayscale image tensor to an RGB image tensor. Each channel in the grayscale image is replicated across the R, G, and B channels of the output.

```APIDOC
## POST /tf.image.grayscaleToRGB

### Description
Converts a grayscale image tensor to an RGB image tensor. Each channel in the grayscale image is replicated across the R, G, and B channels of the output.

### Method
POST

### Endpoint
/tf.image.grayscaleToRGB

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **image** (tf.Tensor|TypedArray|Array) - Required - Tensor of shape `[..., height, width, 1]` or `[..., height, width]`.

### Request Example
```json
{
  "image": "[[[...]]], [[...]]"
}
```

### Response
#### Success Response (200)
- **output** (tf.Tensor) - The RGB image tensor.

#### Response Example
```json
{
  "output": "[[[[...]]]]"
}
```
```

--------------------------------

### tf.encodeString

Source: https://js.tensorflow.org/api/latest/index

Encodes the provided string into bytes using the provided encoding scheme.

```APIDOC
## tf.encodeString (s, encoding?)

### Description
Encodes the provided string into bytes using the provided encoding scheme.

### Method
N/A (Utility function)

### Endpoint
N/A

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
None

### Request Example
```javascript
const stringToEncode = "Hello, TensorFlow.js!";
const encodedBytes = tf.encodeString(stringToEncode, 'utf-8');
console.log(encodedBytes); // Uint8Array
```

### Response
#### Success Response (N/A)
- **Uint8Array** - The encoded bytes.

#### Response Example
```json
// Example output: Uint8Array [ 72, 101, 108, 108, 111, 44, 32, 84, 101, 110, 115, 111, 114, 70, 108, 111, 119, 44, 106, 115, 33 ]
```
```

--------------------------------

### Register Custom Class

Source: https://js.tensorflow.org/api/latest/index

This section details how to register custom classes with TensorFlow.js serialization, enabling custom layers to be saved and loaded.

```APIDOC
## tf.serialization.registerClass

### Description
Registers a class with the serialization map of TensorFlow.js. This is often used for registering custom Layers, so they can be serialized and deserialized.

### Method
`POST`

### Endpoint
`/tfjs/serialization/registerClass`

### Parameters
#### Request Body
- **cls** (SerializableConstructor) - Required - The class to be registered. It must have a public static member called `className` defined and the value must be a non-empty string.
- **pkg** (string) - Optional - The package name for the class.
- **name** (string) - Optional - The specified name for the class.

### Request Example (without package and name)
```javascript
class MyCustomLayer extends tf.layers.Layer {
   static className = 'MyCustomLayer';
   constructor(config) {
     super(config);
   }
}
tf.serialization.registerClass(MyCustomLayer);
```

### Request Example (with package and name)
```javascript
class MyCustomLayer extends tf.layers.Layer {
   static className = 'MyCustomLayer';
   constructor(config) {
     super(config);
   }
}
tf.serialization.registerClass(MyCustomLayer, "Package", "MyLayer");
```

### Response
No specific response body is detailed for this operation, but it modifies the internal serialization map.
```

--------------------------------

### tf.linalg.gramSchmidt

Source: https://js.tensorflow.org/api/latest/index

Gram-Schmidt orthogonalization.

```APIDOC
## POST /tf.linalg.gramSchmidt

### Description
Gram-Schmidt orthogonalization.

### Method
POST

### Endpoint
/tf.linalg.gramSchmidt

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **xs** (tf.Tensor|TypedArray|Array) - Required - The input tensor(s) to orthogonalize.

### Request Example
```json
{
  "xs": [[1, 2], [3, 4]]
}
```

### Response
#### Success Response (200)
- **tensor** (tf.Tensor) - The orthogonalized tensor.

#### Response Example
```json
{
  "tensor": "[Tensor object representation]"
}
```
```

--------------------------------

### Generate Hann Window using TensorFlow.js

Source: https://js.tensorflow.org/api/latest/index

Generates a Hann window (also known as a Hann window), another type of window function used in signal processing. Similar to the Hamming window, it takes the window length as a parameter. This operation is compatible with various backends.

```javascript
tf.signal.hannWindow(10).print();
```

--------------------------------

### Compute 1D Inverse Real Fast Fourier Transform - tf.spectral.irfft

Source: https://js.tensorflow.org/api/latest/index

Computes the 1-dimensional inverse discrete Fourier transform over the inner-most dimension of a real input tensor. This function is available for WebGPU, WebGL, WASM, and CPU execution.

```javascript
const real = tf.tensor1d([1, 2, 3]);
const imag = tf.tensor1d([0, 0, 0]);
const x = tf.complex(real, imag);

x.irfft().print();

```

--------------------------------

### Transpose Tensor with TensorFlow.js

Source: https://js.tensorflow.org/api/latest/index

Transposes a tf.Tensor by permuting its dimensions according to the 'perm' argument. If 'perm' is not provided, it defaults to reversing the dimension order, effectively performing a matrix transpose for 2D tensors. Supports webgpu, webgl, wasm, and cpu.

```javascript
const x = tf.tensor2d([[1, 2], [3, 4]]);

// Transpose without specifying perm (matrix transpose)
x.transpose().print();

// Transpose with specified permutation
// For a 3D tensor [[a, b], [c, d]], permuting dims [1, 0, 2] would swap the first two dims
// Example: tf.tensor3d([[[1, 2], [3, 4]]]).transpose([1, 0, 2]).print();
```

--------------------------------

### Create 2D Tensor with tf.tensor2d (Nested Array)

Source: https://js.tensorflow.org/api/latest/index

Shows how to create a rank-2 tensor (matrix) using `tf.tensor2d` by passing a nested array. This method infers the shape automatically. The output tensor is then printed.

```javascript
// Pass a nested array.
tf.tensor2d([[1, 2], [3, 4]]).print();
```

--------------------------------

### tf.tensor1d

Source: https://js.tensorflow.org/api/latest/index

Creates a rank-1 tf.Tensor with the provided values, shape and dtype.

```APIDOC
## POST /api/v1/tensor1d

### Description
Creates a rank-1 tf.Tensor with the provided values, shape and dtype.
The same functionality can be achieved with tf.tensor(), but in general we recommend using tf.tensor1d() as it makes the code more readable.

### Method
POST

### Endpoint
/api/v1/tensor1d

### Parameters
#### Request Body
- **values** (TypedArray|Array) - Required - The values of the tensor. Can be array of numbers, or a TypedArray.
- **dtype** ('float32'|'int32'|'bool'|'complex64'|'string') - Optional - The data type. Optional 

### Response
#### Success Response (200)
- **tensor1d** (tf.Tensor1D) - The created rank-1 tensor.

#### Response Example
```json
{
  "tensor1d": "<Tensor1D Object>"
}
```
```

--------------------------------

### Create Tensor with Truncated Normal Distribution

Source: https://js.tensorflow.org/api/latest/index

Creates a tensor with values sampled from a truncated normal distribution. Values outside 2 standard deviations from the mean are re-picked. Supports custom shape, mean, standard deviation, data type, and seed. Available on webgpu, webgl, wasm, and cpu.

```javascript
tf.truncatedNormal([2, 2]).print();
```

--------------------------------

### Resize Images Nearest Neighbor

Source: https://js.tensorflow.org/api/latest/index

Resizes a batch of 3D images to a new shape using nearest neighbor interpolation.

```APIDOC
## POST /tf.image.resizeNearestNeighbor

### Description
NearestNeighbor resize a batch of 3D images to a new shape.

### Method
POST

### Endpoint
/tf.image.resizeNearestNeighbor

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **images** (tf.Tensor3D|tf.Tensor4D|TypedArray|Array) - Required - The images, of rank 4 or rank 3, of shape `[batch, height, width, inChannels]`. If rank 3, batch of 1 is assumed.
- **size** ([number, number]) - Required - The new shape `[newHeight, newWidth]` to resize the images to. Each channel is resized individually.
- **alignCorners** (boolean) - Optional - Defaults to False. If true, rescale input by `(new_height - 1) / (height - 1)`, which exactly aligns the 4 corners of images and resized images. If false, rescale by `new_height / height`. Treat similarly the width dimension.
- **halfPixelCenters** (boolean) - Optional - Defaults to `false`. Whether to assume pixels are of half the actual dimensions, and yield more accurate resizes. This flag would also make the floating point coordinates of the top left pixel 0.5, 0.5.

### Request Example
```json
{
  "images": [[[ [0, 0, 0], [1, 1, 1] ]]],
  "size": [10, 10],
  "alignCorners": false,
  "halfPixelCenters": false
}
```

### Response
#### Success Response (200)
- **resizedImages** (tf.Tensor3D|tf.Tensor4D) - The resized images.

#### Response Example
```json
{
  "resizedImages": [[[ [0, 0, 0], [0, 0, 0], ... ]]]
}
```
```

--------------------------------

### tf.matMul

Source: https://js.tensorflow.org/api/latest/index

Computes the dot product of two matrices, A * B.

```APIDOC
## matMul / tf.matMul

### Description
Computes the dot product of two matrices, A * B. These must be matrices.

### Method
Various (e.g., `a.matMul(b)` or `tf.matMul(a, b)`)

### Endpoint
N/A (function call)

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **a** (tf.Tensor|TypedArray|Array) - First matrix in dot product operation.
- **b** (tf.Tensor|TypedArray|Array) - Second matrix in dot product operation.
- **transposeA** (boolean) - Optional. If true, `a` is transposed before multiplication.
- **transposeB** (boolean) - Optional. If true, `b` is transposed before multiplication.

### Request Example
```json
{
  "example": "const a = tf.tensor2d([1, 2], [1, 2]);\nconst b = tf.tensor2d([1, 2, 3, 4], [2, 2]);\n\na.matMul(b).print();  // or tf.matMul(a, b)"
}
```

### Response
#### Success Response (200)
- **result** (tf.Tensor) - The resulting matrix from the multiplication.

#### Response Example
```json
{
  "example": "Output tensor representing the matrix product"
}
```
```

--------------------------------

### Compute 1D Discrete Fourier Transform - tf.spectral.fft

Source: https://js.tensorflow.org/api/latest/index

Computes the 1-dimensional discrete Fourier transform over the inner-most dimension of a complex input tensor. It can be used with various backends like WebGPU, WebGL, WASM, and CPU.

```javascript
const real = tf.tensor1d([1, 2, 3]);
const imag = tf.tensor1d([1, 2, 3]);
const x = tf.complex(real, imag);

x.fft().print();  // tf.spectral.fft(x).print();

```

--------------------------------

### Create 3D Tensor with tf.tensor3d

Source: https://js.tensorflow.org/api/latest/index

This documentation entry describes the `tf.tensor3d` function, which is used to create rank-3 tensors. It mentions that the same functionality can be achieved with `tf.tensor`, but `tf.tensor3d` is recommended for better code readability.

--------------------------------

### Split Strings with tf.string.stringSplit

Source: https://js.tensorflow.org/api/latest/index

Splits elements of a string tensor based on a delimiter. Can optionally skip empty strings. Returns a SparseTensor with indices, values, and shape.

```javascript
const result = tf.string.stringSplit(['hello world',  'a b c'], ' ');
result['indices'].print(); // [[0, 0], [0, 1], [1, 0], [1, 1], [1, 2]]
result['values'].print(); // ['hello', 'world', 'a', 'b', 'c']
result['shape'].print(); // [2, 3]
```

--------------------------------

### tf.inTopKAsync

Source: https://js.tensorflow.org/api/latest/index

Returns whether the targets are in the top K predictions.

```APIDOC
## tf.inTopKAsync

### Description
Returns whether the targets are in the top K predictions.

### Method
`tf.inTopKAsync(predictions, targets, k?)

### Parameters
#### Path Parameters
* None

#### Query Parameters
* None

#### Request Body
* **predictions** (tf.Tensor | TypedArray | Array) - 2-D or higher tf.Tensor with last dimension being at least `k`.
* **targets** (tf.Tensor | TypedArray | Array) - 1-D or higher tf.Tensor.
* **k** (number) - Optional. Number of top elements to look at for computing precision, default to 1.

### Request Example
```javascript
const predictions = tf.tensor2d([[20, 10, 40, 30], [30, 50, -20, 10]]);
const targets = tf.tensor1d([2, 0]);
const precision = await tf.inTopKAsync(predictions, targets);
precision.print();
```

### Response
#### Success Response (200)
* **tf.Tensor** - A boolean tensor indicating whether the targets are in the top K predictions.
```

--------------------------------

### Cropping2D Layer

Source: https://js.tensorflow.org/api/latest/index

A cropping layer for 2D input, typically used for image preprocessing to remove border pixels.

```APIDOC
## POST /tf.layers.cropping2D

### Description
Cropping layer for 2D input (e.g., image). This layer removes slices from the input tensor along the spatial dimensions.

### Method
POST

### Endpoint
`/tf.layers.cropping2D`

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **cropping** (number|number[][]|Array<number[]> | [number, number][]) - Required - Amount to crop inputs by. You can specify a single value to crop all spatial dimensions, or a tuple of two values to crop the first and second dimensions independently.
- **dataFormat** ('channelsFirst'|'channelsLast') - Optional - Format of the data, which determines the ordering of the dimensions in the inputs. Defaults to `channels_last`.

### Request Example
```json
{
  "cropping": [[2, 2], [2, 2]],
  "dataFormat": "channelsLast"
}
```

### Response
#### Success Response (200)
- **Cropping2D** (object) - The Cropping2D layer instance.

#### Response Example
```json
{
  "layer": "Cropping2D"
}
```
```

--------------------------------

### Apply Thresholded ReLU Activation Layer in TensorFlow.js

Source: https://js.tensorflow.org/api/latest/index

Implements the Thresholded Rectified Linear Unit (ReLU) activation function. This layer sets all values below a specified `theta` threshold to zero. It accepts `theta` and standard layer configuration arguments.

```javascript
tf.layers.thresholdedReLU({
  theta: 0.5, // Activation threshold
  inputShape: [28, 28, 1] // Example input shape
});
```

--------------------------------

### Retrieve Tensor Data Asynchronously (JavaScript)

Source: https://js.tensorflow.org/api/latest/index

The `data()` method asynchronously downloads tensor values, returning a promise that resolves with a typed array when the computation is complete. This is the preferred method for non-blocking data retrieval.

```javascript
const data = await tensor.data();
```

--------------------------------

### tf.confusionMatrix

Source: https://js.tensorflow.org/api/latest/index

Computes the confusion matrix from true labels and predicted labels.

```APIDOC
## tf.confusionMatrix

### Description
Computes the confusion matrix from true labels and predicted labels.

### Method
`tf.confusionMatrix(labels, predictions, numClasses)

### Parameters
#### Path Parameters
* None

#### Query Parameters
* None

#### Request Body
* **labels** (tf.Tensor1D | TypedArray | Array) - The target labels, assumed to be 0-based integers for the classes. The shape is `[numExamples]`, where `numExamples` is the number of examples included.
* **predictions** (tf.Tensor1D | TypedArray | Array) - The predicted classes, assumed to be 0-based integers for the classes. Must have the same shape as `labels`.
* **numClasses** (number) - Number of all classes, as an integer. Its value must be larger than the largest element in `labels` and `predictions`.

### Request Example
```javascript
const labels = tf.tensor1d([0, 1, 2, 1, 0], 'int32');
const predictions = tf.tensor1d([0, 2, 2, 1, 0], 'int32');
const numClasses = 3;
const out = tf.math.confusionMatrix(labels, predictions, numClasses);
out.print();
// Expected output matrix:
// [[2, 0, 0],
//  [0, 1, 1],
//  [0, 0, 1]]
```

### Response
#### Success Response (200)
* **tf.Tensor2D** - The confusion matrix.
```

--------------------------------

### tf.util.assert (expr, msg)

Source: https://js.tensorflow.org/api/latest/index

Asserts that the expression is true. Otherwise throws an error with the provided message.

```APIDOC
## tf.util.assert (expr, msg)

### Description
Asserts that the expression is true. Otherwise throws an error with the provided message.

### Method
assert

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **expr** (boolean) - The expression to assert (as a boolean).
- **msg** (() => string) - A function that returns the message to report when throwing an error. We use a function for performance reasons.

### Request Example
```javascript
const x = 2;
tf.util.assert(x === 2, () => 'x is not 2');
```

### Response
#### Success Response (200)
- **void** - This function does not return a value.

#### Response Example
```json
{
  "example": "No return value"
}
```
```

--------------------------------

### Compute Softmax Cross Entropy Loss with TensorFlow.js

Source: https://js.tensorflow.org/api/latest/index

Calculates the softmax cross entropy loss between one-hot encoded labels and predicted logits. Supports optional weights, label smoothing, and reduction types. Uses tf.Tensor inputs and returns a tf.Tensor representing the loss.

```javascript
tf.losses.softmaxCrossEntropy(onehotLabels, logits, weights?, labelSmoothing?, reduction?)
```

--------------------------------

### tf.linspace

Source: https://js.tensorflow.org/api/latest/index

Generates a sequence of evenly spaced numbers.

```APIDOC
## POST /websites/js_tensorflow_api/tf.linspace

### Description
Return an evenly spaced sequence of numbers over the given interval.

### Method
POST

### Endpoint
/websites/js_tensorflow_api/tf.linspace

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **start** (number) - The start value of the sequence.
- **stop** (number) - The end value of the sequence.
- **num** (number) - The number of values to generate.

### Request Example
```json
{
  "start": 0,
  "stop": 9,
  "num": 10
}
```

### Response
#### Success Response (200)
- **sequence** (tf.Tensor1D) - A tensor containing the evenly spaced numbers.

#### Response Example
```json
{
  "sequence": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
}
```
```

--------------------------------

### tf.layers.gru

Source: https://js.tensorflow.org/api/latest/index

Creates a Gated Recurrent Unit (GRU) layer. This layer is a recurrent neural network layer that processes sequences of inputs. It can be configured with various parameters for activation, initialization, regularization, and dropout.

```APIDOC
## tf.layers.gru

### Description
Creates a Gated Recurrent Unit (GRU) layer. This layer is a recurrent neural network layer that processes sequences of inputs.

### Method
`tf.layers.gru(args)`

### Parameters
#### Args (Object)
- **units** (number) - Positive integer, dimensionality of the output space.
- **recurrentActivation** ('elu'|'hardSigmoid'|'linear'|'relu'|'relu6'| 'selu'|'sigmoid'|'softmax'|'softplus'|'softsign'|'tanh'|'swish'|'mish'|'gelu'|'gelu_new') - Activation function to use for the recurrent step. Defaults to `hardSigmoid`. If `null`, no activation is applied.
- **implementation** (number) - Implementation mode, either 1 or 2. Mode 2 is always used for superior performance in TensorFlow.js.
- **activation** ('elu'|'hardSigmoid'|'linear'|'relu'|'relu6'| 'selu'|'sigmoid'|'softmax'|'softplus'|'softsign'|'tanh'|'swish'|'mish'|'gelu'|'gelu_new') - Activation function to use. Defaults to `tanh`. If `null`, no activation is applied.
- **useBias** (boolean) - Whether the layer uses a bias vector.
- **kernelInitializer** (string|tf.initializers.Initializer) - Initializer for the `kernel` weights matrix.
- **recurrentInitializer** (string|tf.initializers.Initializer) - Initializer for the `recurrentKernel` weights matrix.
- **biasInitializer** (string|tf.initializers.Initializer) - Initializer for the bias vector.
- **kernelRegularizer** (string|tf.Regularizer) - Regularizer function applied to the kernel weights matrix.
- **recurrentRegularizer** (string|tf.Regularizer) - Regularizer function applied to the recurrentKernel weights matrix.
- **biasRegularizer** (string|tf.Regularizer) - Regularizer function applied to the bias vector.
- **kernelConstraint** (string|tf.constraints.Constraint) - Constraint function applied to the kernel weights matrix.
- **recurrentConstraint** (string|tf.constraints.Constraint) - Constraint function applied to the recurrentKernel weights matrix.
- **biasConstraint** (string|tf.constraints.Constraint) - Constraint function applied to the bias vector.
- **dropout** (number) - Fraction of the units to drop for the linear transformation of the inputs (0 to 1).
- **recurrentDropout** (number) - Fraction of the units to drop for the linear transformation of the recurrent state (0 to 1).
- **dropoutFunc** (Function) - Added for test DI purpose.
- **cell** (tf.RNNCell|tf.RNNCell[]) - A RNN cell instance.

### Request Example
```javascript
const rnn = tf.layers.gru({units: 8, returnSequences: true});

// Create an input with 10 time steps.
const input = tf.input({shape: [10, 20]});
const output = rnn.apply(input);

console.log(JSON.stringify(output.shape));
// [null, 10, 8]
```

### Response
#### Success Response (200)
Returns a GRU layer instance.

#### Response Example
```json
{
  "class": "GRU",
  "name": "gru_1",
  "units": 8,
  "returnSequences": true
}
```
```

--------------------------------

### Compute Gradients with TensorFlow.js Optimizer

Source: https://js.tensorflow.org/api/latest/index

Computes the gradients of a scalar output of function `f` with respect to a list of trainable variables. If `varList` is not provided, it defaults to all trainable variables. Returns an object containing the scalar value and gradients.

```javascript
tf.train.Optimizer.computeGradients(f, varList?)
```

--------------------------------

### tf.util.shuffleCombo

Source: https://js.tensorflow.org/api/latest/index

Shuffles two arrays in-place the same way using Fisher-Yates algorithm.

```APIDOC
## tf.util.shuffleCombo (array, array2)

### Description
Shuffles two arrays in-place the same way using Fisher-Yates algorithm.

### Method
N/A (Utility function)

### Endpoint
N/A

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
None

### Request Example
```javascript
const a = [1,2,3,4,5];
const b = [11,22,33,44,55];
tf.util.shuffleCombo(a, b);
console.log(a, b);
```

### Response
#### Success Response (N/A)
- **void** - This function shuffles the arrays in-place and does not return a value.

#### Response Example
```json
// Both arrays 'a' and 'b' are modified with the same permutation. Example: a = [3, 1, 5, 2, 4], b = [33, 11, 55, 22, 44]
```
```

--------------------------------

### Conv3D Layer

Source: https://js.tensorflow.org/api/latest/index

Defines a 3D convolution layer that applies a convolution kernel to the input tensor. It supports various parameters for kernel size, strides, padding, data format, dilation rate, activation functions, bias, and initializers/constraints/regularizers for weights.

```APIDOC
## POST /tf.layers.conv3d

### Description
Creates a 3D convolution layer (e.g., spatial convolution over volumes). This layer computes the output of a 3D convolution operation, optionally with bias and activation.

### Method
POST

### Endpoint
`/tf.layers.conv3d`

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **filters** (number) - Required - The dimensionality of the output space (i.e. the number of filters in the convolution).
- **kernelSize** (number|number[]) - Required - The dimensions of the convolution window. If kernelSize is a number, the convolutional window will be square.
- **strides** (number|number[]) - Optional - The strides of the convolution in each dimension. If strides is a number, strides in both dimensions are equal. Specifying any stride value != 1 is incompatible with specifying any `dilationRate` value != 1.
- **padding** ('valid'|'same'|'causal') - Optional - Padding mode. Defaults to 'valid'.
- **dataFormat** ('channelsFirst'|'channelsLast') - Optional - Format of the data, which determines the ordering of the dimensions in the inputs. Defaults to `channels_last`.
- **dilationRate** (number|[number]|[number, number]|[number, number, number]) - Optional - The dilation rate to use for the dilated convolution in each dimension. Should be an integer or array of two or three integers. Currently, specifying any `dilationRate` value != 1 is incompatible with specifying any `strides` value != 1.
- **activation** ('elu'|'hardSigmoid'|'linear'|'relu'|'relu6'| 'selu'|'sigmoid'|'softmax'|'softplus'|'softsign'|'tanh'|'swish'|'mish'|'gelu'|'gelu_new') - Optional - Activation function of the layer. If not specified, none is applied.
- **useBias** (boolean) - Optional - Whether the layer uses a bias vector. Defaults to `true`.
- **kernelInitializer** (string|tf.initializers.Initializer) - Optional - Initializer for the convolutional kernel weights matrix. Defaults to 'glorotUniform'.
- **biasInitializer** (string|tf.initializers.Initializer) - Optional - Initializer for the bias vector. Defaults to 'zeros'.
- **kernelConstraint** (string|tf.constraints.Constraint) - Optional - Constraint for the convolutional kernel weights.
- **biasConstraint** (string|tf.constraints.Constraint) - Optional - Constraint for the bias vector.
- **kernelRegularizer** (string|tf.Regularizer) - Optional - Regularizer function applied to the kernel weights matrix.
- **biasRegularizer** (string|tf.Regularizer) - Optional - Regularizer function applied to the bias vector.
- **activityRegularizer** (string|tf.Regularizer) - Optional - Regularizer function applied to the activation.
- **inputShape** ((null | number)[]) - Optional - If defined, will be used to create an input layer to insert before this layer. Only applicable to input layers.
- **batchInputShape** ((null | number)[]) - Optional - If defined, will be used to create an input layer to insert before this layer. If both `inputShape` and `batchInputShape` are defined, `batchInputShape` will be used. Only applicable to input layers.
- **batchSize** (number) - Optional - If `inputShape` is specified and `batchInputShape` is _not_ specified, `batchSize` is used to construct the `batchInputShape`: `[batchSize, ...inputShape]`.
- **dtype** ('float32'|'int32'|'bool'|'complex64'|'string') - Optional - The data-type for this layer. Defaults to 'float32'. Only applicable to input layers.
- **name** (string) - Optional - Name for this layer.
- **trainable** (boolean) - Optional - Whether the weights of this layer are updatable by `fit`. Defaults to `true`.
- **weights** (tf.Tensor[]) - Optional - Initial weight values of the layer.
- **inputDType** ('float32'|'int32'|'bool'|'complex64'|'string') - Optional - Legacy support. Do not use for new code.

### Request Example
```json
{
  "filters": 32,
  "kernelSize": 3,
  "strides": 1,
  "padding": "same",
  "activation": "relu",
  "inputShape": [128, 128, 128, 1]
}
```

### Response
#### Success Response (200)
- **Conv3D** (object) - The Conv3D layer instance.

#### Response Example
```json
{
  "layer": "Conv3D"
}
```
```

--------------------------------

### tf.tensor6d

Source: https://js.tensorflow.org/api/latest/index

Creates a rank-6 tf.Tensor with the provided values, shape and dtype. Recommended for readability over tf.tensor().

```APIDOC
## tf.tensor6d

### Description
Creates a rank-6 tf.Tensor with the provided values, shape and dtype. This function is recommended for code readability compared to using the generic `tf.tensor()`.

### Method
`tf.tensor6d(values, shape?, dtype?): tf.Tensor6D`

### Parameters
* **values** (TypedArray|Array) - The values of the tensor. Can be a nested array of numbers, a flat array, or a TypedArray.
* **shape** ([number, number, number, number, number, number]) - The shape of the tensor. Optional. If not provided, it is inferred from `values`.
* **dtype** ('float32'|'int32'|'bool'|'complex64'|'string') - The data type of the tensor. Optional.

### Request Example
```javascript
// Pass a nested array.
tf.tensor6d([[[[[[1],[2]],[[3],[4]]],[[[5],[6]],[[7],[8]]]]]]).print();

// Pass a flat array and specify a shape.
tf.tensor6d([1, 2, 3, 4, 5, 6, 7, 8], [1, 1, 2, 2, 2, 1]).print();
```

### Response Example
```json
{
  "example": "tf.Tensor6D"
}
```
```

--------------------------------

### Create Rank-5 Tensor with Flat Array and Shape - tf.tensor5d

Source: https://js.tensorflow.org/api/latest/index

Builds a rank-5 tensor from a flat array and a specified shape, enhancing code clarity.

```javascript
// Pass a flat array and specify a shape.
tf.tensor5d([1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 2, 2, 1]).print();

```

--------------------------------

### tf.initializers.orthogonal

Source: https://js.tensorflow.org/api/latest/index

Initializer that generates a random orthogonal matrix, optionally scaled by a gain factor.

```APIDOC
## tf.initializers.orthogonal (args)

### Description
Initializer that generates a random orthogonal matrix.

### Method
Function

### Endpoint
N/A (Client-side JavaScript function)

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
* **args** (Object) - Configuration object for the initializer.
* **gain** (number) - Optional. Multiplicative factor to apply to the orthogonal matrix. Defaults to 1.
* **seed** (number) - Optional. Random number generator seed.

### Request Example
```json
{
  "args": {
    "gain": 1.5,
    "seed": 103
  }
}
```

### Response
#### Success Response (tf.initializers.Initializer)
Returns an instance of tf.initializers.Initializer.

#### Response Example
```json
{
  "instance": "tf.initializers.Initializer"
}
```

### Reference
Saxe et al., http://arxiv.org/abs/1312.6120
```

--------------------------------

### Compute Gradient of a Function with tf.grad

Source: https://js.tensorflow.org/api/latest/index

Returns a function that computes the gradient of a given function f(x) with respect to its input x. If dy is provided, it computes the gradient of f(x).mul(dy).sum(). f(x) must take a single tensor and return a single tensor.

```javascript
// f(x) = x ^ 2
const f = x => x.square();
// f'(x) = 2x
const g = tf.grad(f);

const x = tf.tensor1d([2, 3]);
g(x).print();
```

```javascript
// f(x) = x ^ 3
const f = x => x.pow(tf.scalar(3, 'int32'));
// f'(x) = 3x ^ 2
const g = tf.grad(f);
// f''(x) = 6x
const gg = tf.grad(g);

const x = tf.tensor1d([2, 3]);
gg(x).print();
```

--------------------------------

### tf.tensor4d

Source: https://js.tensorflow.org/api/latest/index

Creates a rank-4 tf.Tensor with the provided values, shape and dtype. Recommended for readability over tf.tensor().

```APIDOC
## tf.tensor4d

### Description
Creates a rank-4 tf.Tensor with the provided values, shape and dtype. This function is recommended for code readability compared to using the generic `tf.tensor()`.

### Method
`tf.tensor4d(values, shape?, dtype?): tf.Tensor4D`

### Parameters
* **values** (TypedArray|Array) - The values of the tensor. Can be a nested array of numbers, a flat array, or a TypedArray.
* **shape** ([number, number, number, number]) - The shape of the tensor. Optional. If not provided, it is inferred from `values`.
* **dtype** ('float32'|'int32'|'bool'|'complex64'|'string') - The data type of the tensor. Optional.

### Request Example
```javascript
// Pass a nested array.
tf.tensor4d([[[[1], [2]], [[3], [4]]]]).print();

// Pass a flat array and specify a shape.
tf.tensor4d([1, 2, 3, 4], [1, 2, 2, 1]).print();
```

### Response Example
```json
{
  "example": "tf.Tensor4D"
}
```
```

--------------------------------

### Generate Hamming Window using TensorFlow.js

Source: https://js.tensorflow.org/api/latest/index

Generates a Hamming window, a type of weighted window function used in signal processing to reduce spectral leakage. The function takes the desired window length as input. It supports multiple execution backends.

```javascript
tf.signal.hammingWindow(10).print();
```

--------------------------------

### tf.logicalOr (a, b)

Source: https://js.tensorflow.org/api/latest/index

Returns the truth value of `a OR b` element-wise. Supports broadcasting.

```APIDOC
## POST /tf.logicalOr

### Description
Returns the truth value of `a OR b` element-wise. Supports broadcasting.

### Method
POST

### Endpoint
/tf.logicalOr

### Parameters
#### Request Body
- **a** (tf.Tensor|TypedArray|Array) - Required - The first input tensor. Must be of dtype bool.
- **b** (tf.Tensor|TypedArray|Array) - Required - The second input tensor. Must be of dtype bool.

### Request Example
```json
{
  "a": [false, false, true, true],
  "b": [false, true, false, true]
}
```

### Response
#### Success Response (200)
- **result** (tf.Tensor) - The resulting tensor with truth values.

#### Response Example
```json
{
  "result": [false, true, true, true]
}
```
```

--------------------------------

### Wait for Next Frame with tf.nextFrame

Source: https://js.tensorflow.org/api/latest/index

Returns a promise that resolves when a requestAnimationFrame has completed. On Node.js, it uses setImmediate. This provides a simple way to await the next frame.

```javascript
// Example usage:
// await tf.nextFrame();
// console.log('Next frame');
```

--------------------------------

### tf.logSoftmax

Source: https://js.tensorflow.org/api/latest/index

Computes the log softmax of a tensor. It normalizes the input tensor along a specified dimension.

```APIDOC
## POST /tf.logSoftmax

### Description
Computes the log softmax of a tensor. It normalizes the input tensor along a specified dimension.

### Method
POST

### Endpoint
/tf.logSoftmax

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **logits** (tf.Tensor|TypedArray|Array) - Required - The logits array.
- **axis** (number) - Optional - The dimension softmax would be performed on. Defaults to `-1` which indicates the last dimension.

### Request Example
```json
{
  "logits": [[2, 4, 6, 1, 2, 3], [2, 4, 6, 1, 2, 3]],
  "axis": -1
}
```

### Response
#### Success Response (200)
- **output** (tf.Tensor) - The resulting tensor after applying logSoftmax.

#### Response Example
```json
{
  "output": "[[...]]"
}
```
```

--------------------------------

### Compute Euclidean Norm with TensorFlow.js

Source: https://js.tensorflow.org/api/latest/index

Computes the Euclidean norm of scalar, vectors, and matrices using the euclideanNorm function. Optional parameters allow specifying the axis for computation and whether to keep the dimensions of the input. Supports webgpu, webgl, wasm, and cpu.

```javascript
const x = tf.tensor1d([1, 2, 3, 4]);

x.euclideanNorm().print();  // or tf.euclideanNorm(x)
```

--------------------------------

### Map Dataset Elements Asynchronously with JavaScript

Source: https://js.tensorflow.org/api/latest/index

The `mapAsync` method transforms each element of a dataset using an asynchronous function. The transform function should handle disposing intermediate Tensors. It returns a new dataset with transformed elements.

```javascript
const a = 
  tf.data.array([1, 2, 3]).mapAsync(x => new Promise(function(resolve){
    setTimeout(() => {
      resolve(x * x);
    }, Math.random()*1000 + 500);
  }));
console.log(await a.toArray());

```

--------------------------------

### Apply Transforms to Image Tensor - JavaScript

Source: https://js.tensorflow.org/api/latest/index

Applies specified projective transforms to an image tensor. Supports 'nearest' and 'bilinear' interpolation modes, and various fill modes ('constant', 'reflect', 'wrap', 'nearest'). It requires an image tensor and a transform matrix/matrices, with optional interpolation, fill mode, fill value, and output shape. Returns a transformed 4D tensor.

```javascript
tf.image.transform(image, transforms, interpolation?, fillMode?, fillValue?, outputShape?)
```

--------------------------------

### Bitwise AND Operation - JavaScript

Source: https://js.tensorflow.org/api/latest/index

Performs a bitwise AND operation on two input tensors, supporting 'int32' data type. It takes two tensors, `x` and `y`, and returns a new tensor containing the element-wise results of the AND operation. Broadcasting is supported.

```javascript
const x = tf.tensor1d([0, 5, 3, 14], 'int32');
const y = tf.tensor1d([5, 0, 7, 11], 'int32');
tf.bitwiseAnd(x, y).print();
```

--------------------------------

### Resize Images Bilinearly

Source: https://js.tensorflow.org/api/latest/index

Resizes a single 3D image or a batch of 3D images to a new shape using bilinear interpolation.

```APIDOC
## POST /tf.image.resizeBilinear

### Description
Bilinear resize a single 3D image or a batch of 3D images to a new shape.

### Method
POST

### Endpoint
/tf.image.resizeBilinear

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **images** (tf.Tensor3D|tf.Tensor4D|TypedArray|Array) - Required - The images, of rank 4 or rank 3, of shape `[batch, height, width, inChannels]`. If rank 3, batch of 1 is assumed.
- **size** ([number, number]) - Required - The new shape `[newHeight, newWidth]` to resize the images to. Each channel is resized individually.
- **alignCorners** (boolean) - Optional - Defaults to `false`. If true, rescale input by `(new_height - 1) / (height - 1)`, which exactly aligns the 4 corners of images and resized images. If false, rescale by `new_height / height`. Treat similarly the width dimension.
- **halfPixelCenters** (boolean) - Optional - Defaults to `false`. Whether to assume pixel centers are at 0.5, which would make the floating point coordinates of the top left pixel 0.5, 0.5.

### Request Example
```json
{
  "images": [[[ [0, 0, 0], [1, 1, 1] ]]],
  "size": [10, 10],
  "alignCorners": false,
  "halfPixelCenters": false
}
```

### Response
#### Success Response (200)
- **resizedImages** (tf.Tensor3D|tf.Tensor4D) - The resized images.

#### Response Example
```json
{
  "resizedImages": [[[ [0.0, 0.0, 0.0], [0.1, 0.1, 0.1], ... ]]]
}
```
```

--------------------------------

### Batch to Space ND Transformation (JavaScript)

Source: https://js.tensorflow.org/api/latest/index

The `tf.batchToSpaceND()` function reshapes the batch dimension of a tensor and interleaves blocks into spatial dimensions. It is the reverse operation of `tf.spaceToBatchND()` and supports various backends like WebGL, WebGPU, WASM, and CPU.

```javascript
const x = tf.tensor4d([1, 2, 3, 4], [4, 1, 1, 1]);
const blockShape = [2, 2];
const crops = [[0, 0], [0, 0]];

x.batchToSpaceND(blockShape, crops).print();
```

--------------------------------

### Register Custom Layer Class without Package/Name in JavaScript

Source: https://js.tensorflow.org/api/latest/index

Demonstrates registering a custom Layer class with TensorFlow.js serialization without specifying a package name or a custom name. The class will be registered using its default className.

```javascript
class MyCustomLayer extends tf.layers.Layer {
   static className = 'MyCustomLayer';

   constructor(config) {
     super(config);
   }
}
tf.serialization.registerClass(MyCustomLayer);
console.log(tf.serialization.GLOBALCUSTOMOBJECT.get("Custom>MyCustomLayer"));
console.log(tf.serialization.GLOBALCUSTOMNAMES.get(MyCustomLayer));
```

--------------------------------

### Compute Sparse Segment Mean with TensorFlow.js

Source: https://js.tensorflow.org/api/latest/index

Calculates the mean along sparse segments of a tensor. It takes data, indices, and segment IDs as input. The function supports various backends like WebGPU, WebGL, WASM, and CPU.

```javascript
const c = tf.tensor2d([[1,2,3,4], [-1,-2,-3,-4], [6,7,8,9]]);
// Select two rows, one segment.
const result1 = tf.sparse.sparseSegmentMean(c,
                                           tf.tensor1d([0, 1], 'int32'),
                                           tf.tensor1d([0, 0], 'int32'));
result1.print(); // [[0, 0, 0, 0]]

// Select two rows, two segments.
const result2 = tf.sparse.sparseSegmentMean(c,
                                             tf.tensor1d([0, 1], 'int32'),
                                             tf.tensor1d([0, 1], 'int32'));
result2.print(); // [[1, 2, 3, 4], [-1, -2, -3, -4]]

// Select all rows, two segments.
const result3 = tf.sparse.sparseSegmentMean(c,
                                             tf.tensor1d([0, 1, 2], 'int32'),
                                             tf.tensor1d([0, 1, 1], 'int32'));
result3.print(); // [[1.0, 2.0, 3.0, 4.0], [2.5, 2.5, 2.5, 2.5]]
```

--------------------------------

### Configure Input Shape for TensorFlow.js Layers

Source: https://js.tensorflow.org/api/latest/index

Defines how to configure the input shape for TensorFlow.js layers. It supports `batchInputShape` directly or constructs it using `batchSize` and `inputShape`. This is primarily for input layers and affects how data is fed into the model.

```javascript
tf.layers.dense({
  inputShape: [5],
  batchSize: 32, // Example: Used if batchInputShape is not provided
  // Or directly:
  // batchInputShape: [32, 5]
});
```

--------------------------------

### tf.dot

Source: https://js.tensorflow.org/api/latest/index

Computes the dot product of two matrices and/or vectors.

```APIDOC
## dot / tf.dot

### Description
Computes the dot product of two matrices and/or vectors, `t1` and `t2`.

### Method
Various (e.g., `t1.dot(t2)` or `tf.dot(t1, t2)`)

### Endpoint
N/A (function call)

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **t1** (tf.Tensor|TypedArray|Array) - The first tensor in the dot operation.
- **t2** (tf.Tensor|TypedArray|Array) - The second tensor in the dot operation.

### Request Example
```json
{
  "example": "const a = tf.tensor1d([1, 2]);\nconst b = tf.tensor2d([[1, 2], [3, 4]]);\nconst c = tf.tensor2d([[1, 2, 3], [4, 5, 6]]);\n\na.dot(b).print();  // or tf.dot(a, b)\nb.dot(a).print();\nb.dot(c).print();"
}
```

### Response
#### Success Response (200)
- **result** (tf.Tensor) - The result of the dot product operation.

#### Response Example
```json
{
  "example": "Output tensor representing the dot product"
}
```
```

--------------------------------

### TensorFlow.js: Standard Normal Distribution Sampling

Source: https://js.tensorflow.org/api/latest/index

Creates a tensor with values sampled from a standard normal distribution (mean 0, standard deviation 1) using `tf.randomStandardNormal`. It takes shape, optional dtype, and seed as parameters.

```javascript
tf.randomStandardNormal([2, 2]).print();
```

--------------------------------

### Cropping2D Layer

Source: https://js.tensorflow.org/api/latest/index

The Cropping2D layer allows cropping an input tensor at the top, bottom, left, and right sides of an image tensor. It supports different data formats and various cropping configurations.

```APIDOC
## tf.layers.cropping2d

### Description
This layer can crop an input at the top, bottom, left and right side of an image tensor.

### Method
`tf.layers.cropping2d(args)`

### Parameters
#### Arguments (`args`)
- **cropping** (number|[number, number]|[[number, number], [number, number]]) - Required - Dimension of the cropping along the width and the height.
  * If integer: the same symmetric cropping is applied to width and height.
  * If list of 2 integers: interpreted as two different symmetric cropping values for height and width: `[symmetric_height_crop, symmetric_width_crop]`.
  * If a list of 2 lists of 2 integers: interpreted as `[[top_crop, bottom_crop], [left_crop, right_crop]]`
- **dataFormat** ('channelsFirst'|'channelsLast') - Optional - Format of the data, which determines the ordering of the dimensions in the inputs. Defaults to `channels_last`.
- **inputShape** (null | number[]) - Optional - If defined, will be used to create an input layer to insert before this layer.
- **batchInputShape** (null | number[]) - Optional - If defined, will be used to create an input layer to insert before this layer.
- **batchSize** (number) - Optional - If `inputShape` is specified and `batchInputShape` is _not_ specified, `batchSize` is used to construct the `batchInputShape`: `[batchSize, ...inputShape]`
- **dtype** ('float32'|'int32'|'bool'|'complex64'|'string') - Optional - The data-type for this layer. Defaults to 'float32'.
- **name** (string) - Optional - Name for this layer.
- **trainable** (boolean) - Optional - Whether the weights of this layer are updatable by `fit`. Defaults to true.
- **weights** (tf.Tensor[]) - Optional - Initial weight values of the layer.
- **inputDType** ('float32'|'int32'|'bool'|'complex64'|'string') - Optional - Legacy support. Do not use for new code.

### Input Shape
4D tensor with shape:
  * If `dataFormat` is `"channelsLast"`: `[batch, rows, cols, channels]`
  * If `data_format` is `"channels_first"`: `[batch, channels, rows, cols]`.

### Output Shape
4D tensor with shape:
  * If `dataFormat` is `"channelsLast"`: `[batch, croppedRows, croppedCols, channels]`
  * If `dataFormat` is `"channelsFirst"`: `[batch, channels, croppedRows, croppedCols]`.

### Request Example
```javascript
const model = tf.sequential();
model.add(tf.layers.cropping2D({
  cropping:[[2, 2], [2, 2]],
  inputShape: [128, 128, 3]
}));
// Output shape is now [batch, 124, 124, 3]
```

### Returns
Cropping2D layer instance.
```

--------------------------------

### tf.topk

Source: https://js.tensorflow.org/api/latest/index

Finds the values and indices of the `k` largest entries along the last dimension.

```APIDOC
## POST /tf.topk

### Description
Finds the values and indices of the `k` largest entries along the last dimension. If two elements are equal, the lower-index element appears first.

### Method
POST

### Endpoint
/tf.topk

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **x** (tf.Tensor|TypedArray|Array) - Required - 1-D or higher tf.Tensor with last dimension being at least `k`.
- **k** (number) - Optional - Number of top elements to look for along the last dimension.
- **sorted** (boolean) - Optional - If true, the resulting `k` elements will be sorted by the values in descending order.

### Request Example
```json
{
  "x": [[1, 5], [4, 3]],
  "k": 1,
  "sorted": true
}
```

### Response
#### Success Response (200)
- **values** (tf.Tensor) - The top `k` values.
- **indices** (tf.Tensor) - The indices of the top `k` values.

#### Response Example
```json
{
  "values": [5, 4],
  "indices": [1, 0]
}
```
```

--------------------------------

### tf.util.now

Source: https://js.tensorflow.org/api/latest/index

Returns the current high-resolution time in milliseconds relative to an arbitrary time in the past.

```APIDOC
## tf.util.now ()

### Description
Returns the current high-resolution time in milliseconds relative to an arbitrary time in the past. It works across different platforms (node.js, browsers).

### Method
N/A (Utility function)

### Endpoint
N/A

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
None

### Request Example
```javascript
console.log(tf.util.now());
```

### Response
#### Success Response (N/A)
- **number** - The current time in milliseconds.

#### Response Example
```json
123456789.123
```
```

--------------------------------

### AveragePooling3D API

Source: https://js.tensorflow.org/api/latest/index

Performs average pooling operation for 3D data. It downscales the input by a specified pool size across depth, height, and width.

```APIDOC
## AveragePooling3D

### Description
Average pooling operation for 3D data. It downscales the input by factors in depth, height, and width dimensions.

### Method
`tf.layers.averagePooling3d`

### Endpoint
N/A (This is a layer function, not a REST endpoint)

### Parameters
#### Arguments Object (args)
- **poolSize** (number | Array<number>) - Required - Factors by which to downscale in each dimension [depth, height, width].
- **strides** (number | Array<number> | null) - Optional - The size of the stride in each dimension. Defaults to `poolSize`.
- **padding** ('valid' | 'same' | 'causal') - Optional - The padding type to use.
- **dataFormat** ('channelsFirst' | 'channelsLast') - Optional - The data format to use.
- **inputShape** (Array<number | null>) - Optional - Used to create an input layer if not already defined.
- **batchInputShape** (Array<number | null>) - Optional - Used to create an input layer if not already defined.
- **batchSize** (number) - Optional - Used to construct `batchInputShape` if `inputShape` is provided.
- **dtype** ('float32' | 'int32' | 'bool' | 'complex64' | 'string') - Optional - The data-type for this layer. Defaults to 'float32'.
- **name** (string) - Optional - Name for this layer.
- **trainable** (boolean) - Optional - Whether the weights of this layer are updatable. Defaults to true.
- **weights** (Array<tf.Tensor>) - Optional - Initial weight values of the layer.
- **inputDType** ('float32' | 'int32' | 'bool' | 'complex64' | 'string') - Optional - Legacy support, do not use for new code.

### Request Example
```json
{
  "poolSize": [2, 2, 2],
  "strides": [2, 2, 2],
  "padding": "valid",
  "dataFormat": "channelsLast"
}
```

### Response
#### Success Response (Output Shape)
- **Output Shape** (5D tensor) - Shape depends on `dataFormat` and pooling parameters.
  - If `dataFormat='channelsLast'`: `[batchSize, pooledDepths, pooledRows, pooledCols, channels]`
  - If `dataFormat='channelsFirst'`: `[batchSize, channels, pooledDepths, pooledRows, pooledCols]`

#### Response Example
(Output shape depends on input and parameters, not a fixed JSON structure)
```

--------------------------------

### tf.initializers.identity

Source: https://js.tensorflow.org/api/latest/index

Initializer that generates the identity matrix. This is intended for use with square 2D matrices.

```APIDOC
## tf.initializers.identity (args)

### Description
Initializer that generates the identity matrix. Only use for square 2D matrices.

### Method
Function

### Endpoint
N/A (Client-side JavaScript function)

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
* **args** (Object) - Configuration object for the initializer.
* **gain** (number) - Optional. Multiplicative factor to apply to the identity matrix.

### Request Example
```json
{
  "args": {
    "gain": 2
  }
}
```

### Response
#### Success Response (tf.initializers.Initializer)
Returns an instance of tf.initializers.Initializer.

#### Response Example
```json
{
  "instance": "tf.initializers.Initializer"
}
```
```

--------------------------------

### tf.initializers.randomUniform

Source: https://js.tensorflow.org/api/latest/index

Initializer that generates random values drawn from a uniform distribution within a specified range.

```APIDOC
## tf.initializers.randomUniform (args)

### Description
Initializer that generates random values initialized to a uniform distribution. Values will be distributed uniformly between the configured minval and maxval.

### Method
Function

### Endpoint
N/A (Client-side JavaScript function)

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
* **args** (Object) - Configuration object for the initializer.
* **minval** (number) - Optional. Lower bound of the range of random values to generate. Defaults to -1.
* **maxval** (number) - Optional. Upper bound of the range of random values to generate. Defaults to 1.
* **seed** (number) - Optional. Used to seed the random generator.

### Request Example
```json
{
  "args": {
    "minval": -0.5,
    "maxval": 0.5,
    "seed": 105
  }
}
```

### Response
#### Success Response (tf.initializers.Initializer)
Returns an instance of tf.initializers.Initializer.

#### Response Example
```json
{
  "instance": "tf.initializers.Initializer"
}
```
```

--------------------------------

### Fill empty rows in a sparse tensor using TensorFlow.js

Source: https://js.tensorflow.org/api/latest/index

Fills empty rows in a sparse tensor with a specified default value. The operation takes indices, values, dense shape, and a default value as input. It returns an object containing the modified sparse tensor's indices, values, an indicator for empty rows, and a reverse index map for backpropagation. This is useful for ensuring all rows in a sparse tensor have at least one entry.

```javascript
const result = tf.sparse.sparseFillEmptyRows(
   [[0, 0], [1, 0], [1, 3], [1, 4], [3, 2], [3, 3]],
   [0, 10, 13, 14, 32, 33], [5, 6], -1);
console.log(result);
result['outputIndices'].print(); // [[0, 0], [1, 0], [1, 3], [1, 4],
                                  //  [2, 0], [3, 2], [3, 3], [4, 0]]
result['outputValues'].print(); // [0, 10, 13, 14,-1, 32, 33, -1]
result['emptyRowIndicator'].print(); // [false, false, true, false, true]
result['reverseIndexMap'].print(); // [0, 1, 2, 3, 5, 6]
```

--------------------------------

### tf.searchSorted

Source: https://js.tensorflow.org/api/latest/index

Searches for where a value would go in a sorted sequence. This operation returns the bucket-index for each value.

```APIDOC
## POST /tf.searchSorted

### Description
Searches for where a value would go in a sorted sequence. This operation returns the bucket-index for each value. The `side` argument controls which index is returned if a value lands exactly on an edge.

### Method
POST

### Endpoint
/tf.searchSorted

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **sortedSequence** (tf.Tensor|TypedArray|Array) - Required - N-D. Sorted sequence.
- **values** (tf.Tensor|TypedArray|Array) - Required - N-D. Search values.
- **side** ('left'|'right') - Optional - Defaults to 'left'. 'left' corresponds to lower bound and 'right' to upper bound.

### Request Example
```json
{
  "sortedSequence": [[0., 3., 8., 9., 10.], [1., 2., 3., 4., 5.]],
  "values": [[9.8, 2.1, 4.3], [0.1, 6.6, 4.5]],
  "side": "left"
}
```

### Response
#### Success Response (200)
- **result** (tf.Tensor) - The tensor containing the indices.

#### Response Example
```json
{
  "result": [[4, 1, 2], [0, 5, 4]]
}
```
```

--------------------------------

### tf.image.grayscaleToRGB

Source: https://js.tensorflow.org/api/latest/index

Converts images from grayscale format to RGB format. This is useful for preparing grayscale images for models that expect RGB input.

```APIDOC
## tf.image.grayscaleToRGB

### Description
Converts images from grayscale to RGB format.

### Method
N/A (Function)

### Endpoint
N/A (Function)

### Parameters
#### Path Parameters
N/A

#### Query Parameters
N/A

#### Request Body
*   **image** (tf.Tensor2D|tf.Tensor3D|tf.Tensor4D|tf.Tensor5D| tf.Tensor6D|TypedArray|Array) - Required - A grayscale tensor to convert. The `image`'s last dimension must be size 1 with at least a two-dimensional shape.

### Request Example
```json
{
  "image": "tf.Tensor (Grayscale)"
}
```

### Response
#### Success Response (200)
*   **tf.Tensor** - The converted RGB tensor.

#### Response Example
```json
{
  "output_image": "tf.Tensor (RGB)"
}
```
```

--------------------------------

### tf.image.nonMaxSuppressionWithScore

Source: https://js.tensorflow.org/api/latest/index

Performs non-maximum suppression (NMS) on bounding boxes, returning both the selected boxes and their scores.

```APIDOC
## tf.image.nonMaxSuppressionWithScore

### Description
Performs non maximum suppression of bounding boxes based on iou (intersection over union).

### Method
N/A (Function)

### Endpoint
N/A (Function)

### Parameters
#### Path Parameters
N/A

#### Query Parameters
N/A

#### Request Body
*   **boxes** (tf.Tensor2D|TypedArray|Array) - Required - A 2D tensor of shape `[numBoxes, 4]`. Each entry is `[y1, x1, y2, x2]`, where `(y1, x1)` and `(y2, x2)` are the corners of the bounding box.
*   **scores** (tf.Tensor1D|TypedArray|Array) - Required - A 1D tensor providing the box scores of shape `[numBoxes]`.
*   **maxOutputSize** (number) - Required - The maximum number of boxes to be selected.
*   **iouThreshold** (number) - Optional - A float representing the threshold for deciding whether boxes overlap too much with respect to IOU. Must be between [0, 1]. Defaults to 0.5 (50% box overlap).
*   **scoreThreshold** (number) - Optional - A threshold for deciding when to remove boxes based on score. Defaults to -inf, which means any score is accepted.
*   **softNmsSigma** (number) - Optional - Sigma parameter for soft-NMS. If set to 0, it defaults to hard-NMS.

### Request Example
```json
{
  "boxes": "tf.Tensor2D",
  "scores": "tf.Tensor1D",
  "maxOutputSize": 10,
  "iouThreshold": 0.5,
  "scoreThreshold": 0.1,
  "softNmsSigma": 0.5
}
```

### Response
#### Success Response (200)
*   **{[name: string]: tf.Tensor}** - An object containing the selected indices and their corresponding scores.

#### Response Example
```json
{
  "selectedIndices": "tf.Tensor1D",
  "selectedScores": "tf.Tensor1D"
}
```
```

--------------------------------

### Compute Square of Tensor Elements (JavaScript)

Source: https://js.tensorflow.org/api/latest/index

Computes the element-wise square of a tensor (x^2). Supports CPU, WebGL, WebGPU, and WASM backends. Takes a tensor, TypedArray, or Array as input and returns a tensor.

```javascript
const x = tf.tensor1d([1, 2, Math.sqrt(2), -1]);

x.square().print();  // or tf.square(x)
```

--------------------------------

### tf.sub

Source: https://js.tensorflow.org/api/latest/index

Subtracts two tf.Tensors element-wise, A - B. Supports broadcasting.

```APIDOC
## tf.sub

### Description
Subtracts two tf.Tensors element-wise, A - B. Supports broadcasting.

### Method
`tf.sub(a, b)`

### Parameters
* **a** (tf.Tensor | TypedArray | Array) - The first tf.Tensor to subtract from.
* **b** (tf.Tensor | TypedArray | Array) - The second tf.Tensor to be subtracted. Must have the same dtype as `a`.

### Request Example
```javascript
const a = tf.tensor1d([10, 20, 30, 40]);
const b = tf.tensor1d([1, 2, 3, 4]);

a.sub(b).print();  // or tf.sub(a, b)
```

### Response
#### Success Response (200)
* **tf.Tensor**: The resulting tensor after subtraction.

#### Response Example
```json
{
  "data": [9, 18, 27, 36],
  "shape": [4],
  "dtype": "float32"
}
```
```

--------------------------------

### tf.moments

Source: https://js.tensorflow.org/api/latest/index

Calculates the mean and variance of a tensor. The mean and variance are computed by aggregating the contents of the tensor across specified axes.

```APIDOC
## POST /tf.moments

### Description
Calculates the mean and variance of a tensor. The mean and variance are computed by aggregating the contents of the tensor across specified axes. If the tensor is 1-D and `axes = [0]`, this is just the mean and variance of a vector.

### Method
POST

### Endpoint
/tf.moments

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **x** (tf.Tensor|TypedArray|Array) - Required - The input tensor.
- **axis** (number|number[]) - Optional - The dimension(s) along which to compute mean and variance. By default, it reduces all dimensions.
- **keepDims** (boolean) - Optional - If true, the moments will have the same dimensionality as the input.

### Request Example
```json
{
  "x": [[1, 2, 3], [4, 5, 6]],
  "axis": [0, 1],
  "keepDims": false
}
```

### Response
#### Success Response (200)
- **mean** (tf.Tensor) - The computed mean of the tensor.
- **variance** (tf.Tensor) - The computed variance of the tensor.

#### Response Example
```json
{
  "mean": "[[...]]",
  "variance": "[[...]]"
}
```
```

--------------------------------

### Stack Tensors using TensorFlow.js

Source: https://js.tensorflow.org/api/latest/index

Stacks a list of tensors into a single tensor with an increased rank. This TensorFlow.js function concatenates tensors along a new axis. The input tensors must have the same shape and dtype. Supports multiple backends.

```javascript
const a = tf.tensor1d([1, 2]);
const b = tf.tensor1d([3, 4]);
const c = tf.tensor1d([5, 6]);
tf.stack([a, b, c]).print();

```

--------------------------------

### Convert tf.TensorBuffer to tf.Tensor (JavaScript)

Source: https://js.tensorflow.org/api/latest/index

The `toTensor()` method converts a mutable tf.TensorBuffer into an immutable tf.Tensor object.

```javascript
const tensor = buffer.toTensor();
```

--------------------------------

### Create Diagonal Tensor from Vector (JavaScript)

Source: https://js.tensorflow.org/api/latest/index

Constructs a diagonal tensor from a 1D tensor. The output tensor's rank is twice the input tensor's rank, padded with zeros. Supported backends include webgpu, webgl, wasm, and cpu.

```javascript
const x = tf.tensor1d([1, 2, 3, 4]);

tf.diag(x).print();
```

--------------------------------

### tf.mul

Source: https://js.tensorflow.org/api/latest/index

Multiplies two tf.Tensors element-wise, A * B. Supports broadcasting.

```APIDOC
## tf.mul

### Description
Multiplies two tf.Tensors element-wise, A * B. Supports broadcasting.

### Method
`tf.mul(a, b)`

### Parameters
* **a** (tf.Tensor | TypedArray | Array) - The first tensor to multiply.
* **b** (tf.Tensor | TypedArray | Array) - The second tensor to multiply. Must have the same dtype as `a`.

### Request Example
```javascript
const a = tf.tensor1d([1, 2, 3, 4]);
const b = tf.tensor1d([2, 3, 4, 5]);

a.mul(b).print();  // or tf.mul(a, b)
```

### Response
#### Success Response (200)
* **tf.Tensor**: The resulting tensor after multiplication.

#### Response Example
```json
{
  "data": [2, 6, 12, 20],
  "shape": [4],
  "dtype": "float32"
}
```
```

--------------------------------

### tf.floorDiv

Source: https://js.tensorflow.org/api/latest/index

Divides two tf.Tensors element-wise, A / B. Supports broadcasting. The result is rounded with the floor function.

```APIDOC
## tf.floorDiv

### Description
Divides two tf.Tensors element-wise, A / B. Supports broadcasting. The result is rounded with the floor function.

### Method
`tf.floorDiv(a, b)` or `a.floorDiv(b)`

### Parameters
- **a** (tf.Tensor|TypedArray|Array) - The first tensor as the numerator.
- **b** (tf.Tensor|TypedArray|Array) - The second tensor as the denominator. Must have the same dtype as `a`.

### Request Example
```javascript
const a = tf.tensor1d([1, 4, 9, 16]);
const b = tf.tensor1d([1, 2, 3, 4]);

a.floorDiv(b).print();  // or tf.div(a, b)
```

### Response
#### Success Response (tf.Tensor)
- **result** (tf.Tensor) - The element-wise floor division of a and b.

#### Response Example
```json
// Example output for a.floorDiv(b):
// Tensor
//    [1, 2, 3, 4]
```
```

--------------------------------

### Create Rank-4 Tensor with Flat Array and Shape - tf.tensor4d

Source: https://js.tensorflow.org/api/latest/index

Constructs a rank-4 tensor by providing a flat array and its shape. This approach promotes clearer code.

```javascript
// Pass a flat array and specify a shape.
tf.tensor4d([1, 2, 3, 4], [1, 2, 2, 1]).print();

```

--------------------------------

### Compute element-wise hyperbolic cosine using TensorFlow.js

Source: https://js.tensorflow.org/api/latest/index

Computes the hyperbolic cosine (cosh) of the input tensor element-wise. The `cosh` function takes a tensor `x` and returns a new tensor with the hyperbolic cosine values.

```javascript
const x = tf.tensor1d([0, 1, -1, .7]);

x.cosh().print();  // or tf.cosh(x)
```

--------------------------------

### Standard Normal Distribution Sampling

Source: https://js.tensorflow.org/api/latest/index

Creates a tensor with values sampled from a normal distribution with mean 0 and standard deviation 1.

```APIDOC
## POST /randomStandardNormal

### Description
Creates a tf.Tensor with values sampled from a normal distribution. The generated values will have mean 0 and standard deviation 1.

### Method
POST

### Endpoint
`/randomStandardNormal`

### Parameters
#### Request Body
- **shape** (number[]) - Required - An array of integers defining the output tensor shape.
- **dtype** ('float32'|'int32') - Optional - The data type of the output. Defaults to float32.
- **seed** (number) - Optional - The seed for the random number generator.

### Request Example
```json
{
  "shape": [2, 2]
}
```

### Response
#### Success Response (200)
- **result** (tf.Tensor) - A tensor with values sampled from the standard normal distribution.

#### Response Example
```json
{
  "result": [[0.1, -0.5], [1.2, -0.8]]
}
```
```

--------------------------------

### Apply Gradients with TensorFlow.js Optimizer

Source: https://js.tensorflow.org/api/latest/index

Updates variables using computed gradients. Accepts a mapping of variable names to their gradient values. This method modifies variables in place and returns void.

```javascript
tf.train.Optimizer.applyGradients(variableGradients)
```

--------------------------------

### tf.decodeString

Source: https://js.tensorflow.org/api/latest/index

Decodes the provided bytes into a string using the provided encoding scheme.

```APIDOC
## tf.decodeString (bytes, encoding?)

### Description
Decodes the provided bytes into a string using the provided encoding scheme.

### Method
N/A (Utility function)

### Endpoint
N/A

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
None

### Request Example
```javascript
// Assuming 'bytes' is a Uint8Array containing UTF-8 encoded data
const decodedString = tf.decodeString(bytes, 'utf-8');
console.log(decodedString);
```

### Response
#### Success Response (N/A)
- **string** - The decoded string.

#### Response Example
```json
"Hello, TensorFlow.js!"
```
```

--------------------------------

### Rotate Image Tensor with Offset - JavaScript

Source: https://js.tensorflow.org/api/latest/index

Rotates an input image tensor counter-clockwise with an optional offset center of rotation. This function is available on CPU, WebGL, and WASM backends. It takes a 4D tensor, rotation amount in radians, an optional fill value, and an optional center of rotation as input. It returns a rotated 4D tensor.

```javascript
tf.image.rotateWithOffset(image, radians, fillValue?, center?)
```

--------------------------------

### tf.divNoNan

Source: https://js.tensorflow.org/api/latest/index

Divides two tf.Tensors element-wise, A / B. Supports broadcasting. Returns 0 if the denominator is 0.

```APIDOC
## tf.divNoNan

### Description
Divides two tf.Tensors element-wise, A / B. Supports broadcasting. Returns 0 if the denominator is 0.

### Method
`tf.divNoNan(a, b)` or `a.divNoNan(b)`

### Parameters
- **a** (tf.Tensor|TypedArray|Array) - The first tensor as the numerator.
- **b** (tf.Tensor|TypedArray|Array) - The second tensor as the denominator. Must have the same dtype as `a`.

### Request Example
```javascript
const a = tf.tensor1d([1, 4, 9, 16]);
const b = tf.tensor1d([1, 2, 3, 4]);
const c = tf.tensor1d([0, 0, 0, 0]);

a.divNoNan(b).print();  // or tf.divNoNan(a, b)
a.divNoNan(c).print();  // or tf.divNoNan(a, c)
```

### Response
#### Success Response (tf.Tensor)
- **result** (tf.Tensor) - The element-wise division of a and b, with 0 for division by zero.

#### Response Example
```json
// Example output for a.divNoNan(b):
// Tensor
//    [1, 2, 3, 4]
// Example output for a.divNoNan(c):
// Tensor
//    [0, 0, 0, 0]
```
```

--------------------------------

### Perform element-wise less than or equal to comparison

Source: https://js.tensorflow.org/api/latest/index

This function performs an element-wise less than or equal to comparison between two tensors. It supports broadcasting and requires input tensors to have the same data type. The output is a boolean tensor indicating the truth value of the comparison.

```javascript
const a = tf.tensor1d([1, 2, 3]);
const b = tf.tensor1d([2, 2, 2]);

a.lessEqual(b).print();
```

--------------------------------

### Multi-Cell LSTM Computation - JavaScript

Source: https://js.tensorflow.org/api/latest/index

Computes the next states and outputs for a stack of LSTMCells. Each cell's output is fed as input to the next cell in the stack. Requires an array of LSTMCell functions, input data, an array of previous cell states, and an array of previous hidden outputs. Returns arrays of new cell states and new cell outputs.

```javascript
tf.multiRNNCell(lstmCells, data, c, h)
```

--------------------------------

### Define an Activation Layer in TensorFlow.js

Source: https://js.tensorflow.org/api/latest/index

Defines an activation layer in TensorFlow.js. This layer applies an element-wise activation function to its input. Supported activation functions include 'elu', 'hardSigmoid', 'linear', 'relu', 'relu6', 'selu', 'sigmoid', 'softmax', 'softplus', 'softsign', 'tanh', 'swish', 'mish', 'gelu', and 'gelu_new'.

```javascript
tf.layers.activation({
  activation: 'relu'
});
```

--------------------------------

### Create Rank-6 Tensor with Flat Array and Shape - tf.tensor6d

Source: https://js.tensorflow.org/api/latest/index

Constructs a rank-6 tensor using a flat array and an explicit shape, leading to more readable code.

```javascript
// Pass a flat array and specify a shape.
tf.tensor6d([1, 2, 3, 4, 5, 6, 7, 8], [1, 1, 2, 2, 2, 1]).print();

```

--------------------------------

### Register Custom Layer Class with Specified Package Name in JavaScript

Source: https://js.tensorflow.org/api/latest/index

Demonstrates registering a custom Layer class with only a specified package name ('Package') in TensorFlow.js serialization. The class name will be derived from the `className` static property of the custom class.

```javascript
class MyCustomLayer extends tf.layers.Layer {
   static className = 'MyCustomLayer';

   constructor(config) {
     super(config);
   }
}
tf.serialization.registerClass(MyCustomLayer, "Package");
console.log(tf.serialization.GLOBALCUSTOMOBJECT
.get("Package>MyCustomLayer"));
console.log(tf.serialization.GLOBALCUSTOMNAMES
.get(MyCustomLayer));
```

--------------------------------

### Override Gradient Computation with tf.customGrad

Source: https://js.tensorflow.org/api/latest/index

Allows overriding the gradient computation of a function `f`. It takes a function that returns a `{value, gradFunc}` object, where `gradFunc` computes the custom gradients. Tensors needed for the gradient can be saved using the `save` function. This is useful for defining custom operations. Supported on webgpu, webgl, wasm, and cpu.

```javascript
const customOp = tf.customGrad((x, save) => {
   save([x]);
   return {
     value: x.square(),
     gradFunc: (dy, saved) => [dy.mul(saved[0].abs())]
   };
});

const x = tf.tensor1d([-1, -2, 3]);
const dx = tf.grad(x => customOp(x));

console.log(`f(x):`);
customOp(x).print();
console.log(`f'(x):`);
dx(x).print();

```

--------------------------------

### Create 2D Tensor with tf.tensor2d (Flat Array and Shape)

Source: https://js.tensorflow.org/api/latest/index

Illustrates creating a rank-2 tensor using `tf.tensor2d` with a flat array and explicitly defining the shape. This approach is useful when the data is in a flattened format but needs to be represented as a matrix.

```javascript
// Pass a flat array and specify a shape.
tf.tensor2d([1, 2, 3, 4], [2, 2]).print();
```

--------------------------------

### Perform element-wise less than comparison

Source: https://js.tensorflow.org/api/latest/index

This function performs an element-wise less than comparison between two tensors. It supports broadcasting and requires input tensors to have the same data type. The output is a boolean tensor indicating the truth value of the comparison.

```javascript
const a = tf.tensor1d([1, 2, 3]);
const b = tf.tensor1d([2, 2, 2]);

a.less(b).print();
```

--------------------------------

### tf.outerProduct

Source: https://js.tensorflow.org/api/latest/index

Computes the outer product of two vectors.

```APIDOC
## outerProduct / tf.outerProduct

### Description
Computes the outer product of two vectors, `v1` and `v2`.

### Method
`tf.outerProduct(v1, v2)`

### Endpoint
N/A (function call)

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **v1** (tf.Tensor1D|TypedArray|Array) - The first vector in the outer product operation.
- **v2** (tf.Tensor1D|TypedArray|Array) - The second vector in the outer product operation.

### Request Example
```json
{
  "example": "const a = tf.tensor1d([1, 2, 3]);\nconst b = tf.tensor1d([3, 4, 5]);\n\ntf.outerProduct(a, b).print();"
}
```

### Response
#### Success Response (200)
- **result** (tf.Tensor2D) - The resulting 2D tensor from the outer product.

#### Response Example
```json
{
  "example": "Output tensor representing the outer product"
}
```
```

--------------------------------

### tf.maximum

Source: https://js.tensorflow.org/api/latest/index

Returns the max of a and b (`a > b ? a : b`) element-wise. Supports broadcasting.

```APIDOC
## tf.maximum

### Description
Returns the max of a and b (`a > b ? a : b`) element-wise. Supports broadcasting.

### Method
`tf.maximum(a, b)` or `a.maximum(b)`

### Parameters
- **a** (tf.Tensor|TypedArray|Array) - The first tensor.
- **b** (tf.Tensor|TypedArray|Array) - The second tensor. Must have the same type as `a`.

### Request Example
```javascript
const a = tf.tensor1d([1, 4, 3, 16]);
const b = tf.tensor1d([1, 2, 9, 4]);

a.maximum(b).print();  // or tf.maximum(a, b)
```

### Response
#### Success Response (tf.Tensor)
- **result** (tf.Tensor) - The element-wise maximum of a and b.

#### Response Example
```json
// Example output for a.maximum(b):
// Tensor
//    [1, 4, 9, 16]
```
```

--------------------------------

### Generate Batch Predictions with tf.Model.predictOnBatch

Source: https://js.tensorflow.org/api/latest/index

Returns predictions for a single batch of samples using a TensorFlow.js model. This method is suitable for processing one batch of data at a time. It accepts input samples as a Tensor or an array of Tensors.

```javascript
const model = tf.sequential({
   layers: [tf.layers.dense({units: 1, inputShape: [10]})]
});
model.predictOnBatch(tf.ones([8, 10])).print();
```

--------------------------------

### Normal Distribution Sampling

Source: https://js.tensorflow.org/api/latest/index

Creates a tensor with values sampled from a normal distribution.

```APIDOC
## POST /randomNormal

### Description
Creates a tf.Tensor with values sampled from a normal distribution.

### Method
POST

### Endpoint
`/randomNormal`

### Parameters
#### Request Body
- **shape** (number[]) - Required - An array of integers defining the output tensor shape.
- **mean** (number) - Optional - The mean of the normal distribution. Defaults to 0.
- **stdDev** (number) - Optional - The standard deviation of the normal distribution. Defaults to 1.
- **dtype** ('float32'|'int32') - Optional - The data type of the output. Defaults to float32.
- **seed** (number) - Optional - The seed for the random number generator.

### Request Example
```json
{
  "shape": [2, 2]
}
```

### Response
#### Success Response (200)
- **result** (tf.Tensor) - A tensor with values sampled from the normal distribution.

#### Response Example
```json
{
  "result": [[0.1, -0.5], [1.2, -0.8]]
}
```
```

--------------------------------

### Average Pooling 3D

Source: https://js.tensorflow.org/api/latest/index

Computes the 3D average pooling of an input tensor. Supports various data formats and rounding modes.

```APIDOC
## POST /tf.avgPool3d

### Description
Computes the 3D average pooling. This operation supports the following backends: webgpu, webgl, wasm, cpu.

### Method
POST

### Endpoint
/tf.avgPool3d

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **x** (tf.Tensor4D|tf.Tensor5D|TypedArray|Array) - The input tensor, of rank 5 or rank 4 of shape `[batch, depth, height, width, inChannels]`.
- **filterSize** ([number, number, number]|number) - The filter size: `[filterDepth, filterHeight, filterWidth]`. If `filterSize` is a single number, then `filterDepth == filterHeight == filterWidth`.
- **strides** ([number, number, number]|number) - The strides of the pooling: `[strideDepth, strideHeight, strideWidth]`. If `strides` is a single number, then `strideDepth == strideHeight == strideWidth`.
- **pad** ('valid'|'same'|number) - The type of padding algorithm. `same` and stride 1: output will be of same size as input, regardless of filter size. `valid`: output will be smaller than input if filter is larger than 1*1x1. For more info, see this guide: https://www.tensorflow.org/api_docs/python/tf/nn/convolution
- **dimRoundingMode** ('floor'|'round'|'ceil') - A string from: 'ceil', 'round', 'floor'. If none is provided, it will default to truncate. Optional.
- **dataFormat** ('NDHWC'|'NCDHW') - An optional string from: "NDHWC", "NCDHW". Defaults to "NDHWC". Specify the data format of the input and output data. With the default format "NDHWC", the data is stored in the order of: [batch, depth, height, width, channels]. Only "NDHWC" is currently supported. Optional.

### Request Example
```json
{
  "x": [
    [[[[1], [2]], [[3], [4]]],
    [[[5], [6]], [[7], [8]]]
  ],
  "shape": [1, 2, 2, 2, 1],
  "filterSize": 2,
  "strides": 1,
  "pad": "valid"
}
```

### Response
#### Success Response (200)
- **tensor** (tf.Tensor4D|tf.Tensor5D) - The output tensor after 3D average pooling.

#### Response Example
```json
{
  "tensor": [
    [[[[1],[2]],[[3],[4]]]]
  ]
}
```
```

--------------------------------

### Perform element-wise logical AND operation

Source: https://js.tensorflow.org/api/latest/index

This function performs an element-wise logical AND operation between two boolean tensors. It supports broadcasting and requires both input tensors to be of boolean dtype. The output is a boolean tensor indicating the result of the AND operation.

```javascript
const a = tf.tensor1d([false, false, true, true], 'bool');
const b = tf.tensor1d([false, true, false, true], 'bool');

a.logicalAnd(b).print();
```

--------------------------------

### Compute Logical AND of Tensor Elements

Source: https://js.tensorflow.org/api/latest/index

Computes the logical AND of elements across specified dimensions of a boolean tensor. The rank of the tensor is reduced unless keepDims is true. Supports various backends like webgpu, webgl, wasm, and cpu.

```javascript
const x = tf.tensor1d([1, 1, 1], 'bool');

x.all().print();  // or tf.all(x)
```

```javascript
const x = tf.tensor2d([1, 1, 0, 0], [2, 2], 'bool');

const axis = 1;
x.all(axis).print();  // or tf.all(x, axis)
```

--------------------------------

### Encode String to Bytes with tf.encodeString

Source: https://js.tensorflow.org/api/latest/index

Encodes a string into a Uint8Array using a specified encoding. Defaults to 'utf-8' if no encoding is provided. Useful for preparing string data for storage or transmission.

```javascript
const text = "tensorflow";
const encodedBytes = tf.encodeString(text, 'utf-8');
console.log(encodedBytes);
```

--------------------------------

### Configuring Stateful RNN Layers in TensorFlow.js

Source: https://js.tensorflow.org/api/latest/index

This snippet demonstrates how to configure an RNN layer to be stateful in TensorFlow.js. Stateful RNNs reuse computed states between batches, requiring specific settings for batch size and fitting. This is achieved by setting `stateful: true` and defining a fixed `batchInputShape` or `batchShape` for the model, along with `shuffle: false` during `fit`.

```javascript
const model = tf.sequential({
  layers: [
    tf.layers.dense({units: 32, activation: 'relu', inputShape: [10]}),
    tf.layers.lstm({units: 64, stateful: true, batchInputShape: [32, 10, 1]})
  ]
});

// To reset states:
// model.resetStates();

// During fitting:
// model.fit(xs, ys, {shuffle: false});
```

--------------------------------

### Convert Tensor to Pixels (Browser JavaScript)

Source: https://js.tensorflow.org/api/latest/index

Converts a `tf.Tensor` representing pixel values into a `Uint8ClampedArray`, optionally drawing it to an HTML canvas. Handles float32 tensors (assumed range [0-1]) and int32 tensors (assumed range [0-255]). Supports rank-2 tensors (grayscale) and rank-3 tensors (1, 3, or 4 channels for grayscale, RGB, or RGBA respectively).

```javascript
tf.browser.toPixels (img, canvas?) function Source
Draws a tf.Tensor of pixel values to a byte array or optionally a canvas.
When the dtype of the input is 'float32', we assume values in the range [0-1]. Otherwise, when input is 'int32', we assume values in the range [0-255].
Returns a promise that resolves when the canvas has been drawn to.
Parameters:
  * img (tf.Tensor2D|tf.Tensor3D|TypedArray|Array) A rank-2 tensor with shape `[height, width]`, or a rank-3 tensor of shape `[height, width, numChannels]`. If rank-2, draws grayscale. If rank-3, must have depth of 1, 3 or 4. When depth of 1, draws grayscale. When depth of 3, we draw with the first three components of the depth dimension corresponding to r, g, b and alpha = 1. When depth of 4, all four components of the depth dimension correspond to r, g, b, a.
  * canvas (HTMLCanvasElement) The canvas to draw to. Optional 

Returns: Promise<Uint8ClampedArray>
```

--------------------------------

### toArray ()

Source: https://js.tensorflow.org/api/latest/index

Collect all elements of this dataset into an array. This is only suitable for small datasets that fit in memory.

```APIDOC
## toArray ()

### Description
Collect all elements of this dataset into an array. This will succeed only for small datasets that fit in memory. Useful for testing and generally should be avoided if possible.

### Method
toArray

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
None

### Request Example
```javascript
const a = tf.data.array([1, 2, 3, 4, 5, 6]);
console.log(await a.toArray());
```

### Response
#### Success Response (200)
- **Promise<T[]>** - A Promise that resolves to an array containing all elements of the dataset.

#### Response Example
```json
{
  "example": "[1, 2, 3, 4, 5, 6]"
}
```
```

--------------------------------

### Compute Parametric ReLU Element-wise with tf.prelu

Source: https://js.tensorflow.org/api/latest/index

Computes the parametric leaky rectified linear element-wise function. It takes an input tensor `x` and a parametric alpha tensor, returning a tensor with the computed values. Available for webgpu, webgl, wasm, and cpu backends.

```javascript
const x = tf.tensor1d([-1, 2, -3, 4]);
const alpha = tf.scalar(0.1);

x.prelu(alpha).print();  // or tf.prelu(x, alpha)
```

--------------------------------

### Multinomial Distribution Sampling

Source: https://js.tensorflow.org/api/latest/index

Creates a tensor with values drawn from a multinomial distribution.

```APIDOC
## POST /multinomial

### Description
Creates a tf.Tensor with values drawn from a multinomial distribution.

### Method
POST

### Endpoint
`/multinomial`

### Parameters
#### Request Body
- **logits** (tf.Tensor1D|tf.Tensor2D|TypedArray|Array) - Required - 1D array with unnormalized log-probabilities, or 2D array of shape `[batchSize, numOutcomes]`. See the `normalized` parameter.
- **numSamples** (number) - Required - Number of samples to draw for each row slice.
- **seed** (number) - Optional - The seed number.
- **normalized** (boolean) - Optional - Whether the provided `logits` are normalized true probabilities (sum to 1). Defaults to false.

### Request Example
```json
{
  "logits": [0.75, 0.25],
  "numSamples": 3
}
```

### Response
#### Success Response (200)
- **result** (tf.Tensor1D|tf.Tensor2D) - A tensor with sampled values from the multinomial distribution.

#### Response Example
```json
{
  "result": [0, 0, 1]
}
```
```

--------------------------------

### tf.scatterND

Source: https://js.tensorflow.org/api/latest/index

Creates a new tensor by applying sparse updates to individual values or slices within a zero tensor.

```APIDOC
## tf.scatterND

### Description
Creates a new tensor by applying sparse updates to individual values or slices within a zero tensor of the given shape, according to `indices`. This operator is the inverse of the `tf.gatherND()` operator.

### Method
`tf.scatterND(indices, updates, shape)
`

### Parameters
- **indices** (tf.Tensor | TypedArray | Array) - Index tensor, must be of type int32. Specifies the locations within the output tensor to be updated.
- **updates** (tf.Tensor | TypedArray | Array) - The tensor containing the new values to be scattered into the output tensor.
- **shape** (number | number[] | Shape) - The shape of the output tensor.

### Returns
- **tf.Tensor** - A new tensor with sparse updates applied.

### Request Example
```javascript
const indices = tf.tensor2d([4, 3, 1, 7], [4, 1], 'int32');
const updates = tf.tensor1d([9, 10, 11, 12]);
const shape = [8];
tf.scatterND(indices, updates, shape).print() // [0, 11, 0, 10, 9, 0, 0, 12]
```
```

--------------------------------

### Compute QR decomposition of a matrix using TensorFlow.js

Source: https://js.tensorflow.org/api/latest/index

Computes the QR decomposition of a given matrix using Householder transformations. The function returns two matrices, Q and R, where Q is an orthogonal matrix and R is an upper triangular matrix. The operation can be configured to return full-sized Q or only the leading columns.

```javascript
const a = tf.tensor2d([[1, 2], [3, 4]]);
let [q, r] = tf.linalg.qr(a);
console.log('Q');
q.print();
console.log('R');
r.print();
console.log('Orthogonalized');
q.dot(q.transpose()).print()  // should be nearly the identity matrix.
console.log('Reconstructed');
q.dot(r).print(); // should be nearly [[1, 2], [3, 4]];
```

--------------------------------

### RandomWidth Layer for Image Augmentation in TensorFlow.js

Source: https://js.tensorflow.org/api/latest/index

The RandomWidth layer in TensorFlow.js is used for data augmentation during training by randomly adjusting the width of input images. It accepts a 'factor' to control the degree of width variation and supports different interpolation methods. The layer is inactive during inference by default.

```javascript
const layer = tf.layers.randomWidth({
  factor: 0.2
});

const batchedImage = tf.randomNormal([2, 100, 50, 3]); // Batch of 2 images, 100x50 pixels, 3 channels
const augmentedImage = layer.apply(batchedImage);
afterResize.print();
// The output shape will have a width between 40 and 60 pixels.
```

--------------------------------

### tf.clipByValue

Source: https://js.tensorflow.org/api/latest/index

Clips tensor values element-wise to be within a specified range.

```APIDOC
## tf.clipByValue (x, clipValueMin, clipValueMax)

### Description
Clips values element-wise. `max(min(x, clipValueMax), clipValueMin)`.

### Method
`tf.clipByValue` or Tensor.clipByValue

### Parameters
* `x` (tf.Tensor|TypedArray|Array) - The input tensor.
* `clipValueMin` (number) - Lower bound of range to be clipped to.
* `clipValueMax` (number) - Upper bound of range to be clipped to.

### Request Example
```javascript
const x = tf.tensor1d([-1, 2, -3, 4]);

x.clipByValue(-2, 3).print();  // or tf.clipByValue(x, -2, 3)
```

### Response Example
```javascript
// Tensor
//   [-1,
//    2,
//   -2,
//    3]
```
```

--------------------------------

### Compute Short-time Fourier Transform using TensorFlow.js

Source: https://js.tensorflow.org/api/latest/index

Computes the Short-time Fourier Transform (STFT) of a signal, which decomposes a time-series signal into its constituent frequency components over short, overlapping time intervals. It allows specifying frame length, step, FFT length, and an optional window function. Supported on multiple backends.

```javascript
const input = tf.tensor1d([1, 1, 1, 1, 1])
tf.signal.stft(input, 3, 1).print();
```

--------------------------------

### TensorFlow.js: Normal Distribution Sampling

Source: https://js.tensorflow.org/api/latest/index

Generates a tensor with values sampled from a normal (Gaussian) distribution using `tf.randomNormal`. Users can specify shape, mean, standard deviation, data type, and a seed.

```javascript
tf.randomNormal([2, 2]).print();
```

--------------------------------

### Compute Step Function of Tensor Elements (JavaScript)

Source: https://js.tensorflow.org/api/latest/index

Computes the element-wise step function of a tensor: 1 if x > 0, otherwise alpha. Supports CPU, WebGL, WebGPU, and WASM backends. Takes a tensor, TypedArray, or Array and an optional alpha value, returning a tensor.

```javascript
const x = tf.tensor1d([0, 2, -1, -3]);

x.step(.5).print();  // or tf.step(x, .5)
```

--------------------------------

### Equality Operation

Source: https://js.tensorflow.org/api/latest/index

Checks for element-wise equality between two tensors.

```APIDOC
## tf.equal

### Description
Returns the truth value of (a == b) element-wise. Supports broadcasting.

### Method
Not specified (assumed to be a function call)

### Endpoint
N/A

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **a** (tf.Tensor|TypedArray|Array) - Required - The first input tensor.
- **b** (tf.Tensor|TypedArray|Array) - Required - The second input tensor. Must have the same dtype as `a`.

### Request Example
```javascript
const a = tf.tensor1d([1, 2, 3]);
const b = tf.tensor1d([2, 2, 2]);

a.equal(b).print();
// Output: 
// Tensor
//    [false,
//     true,
//     false]
```

### Response
#### Success Response (200)
- **result** (tf.Tensor) - A boolean tensor indicating element-wise equality.
```

--------------------------------

### Apply Sparse Updates to Tensor (JavaScript)

Source: https://js.tensorflow.org/api/latest/index

The tf.scatterND function creates a new tensor by applying sparse updates to a zero tensor based on specified indices. It is the inverse operation of tf.gatherND. This function is primarily supported on the webgpu backend.

```javascript
const indices = tf.tensor2d([4, 3, 1, 7], [4, 1], 'int32');
const updates = tf.tensor1d([9, 10, 11, 12]);
const shape = [8];
tf.scatterND(indices, updates, shape).print() //[0, 11, 0, 10, 9, 0, 0, 12]
```

--------------------------------

### Apply Dense Layer to Each Time Step with TimeDistributed (JavaScript)

Source: https://js.tensorflow.org/api/latest/index

Demonstrates using tf.layers.timeDistributed to apply a Dense layer to each temporal slice of an input sequence. The input shape is specified, and the output shape is shown to include the transformed units. This is useful for processing sequential data where each time step needs independent transformation.

```javascript
const model = tf.sequential();
model.add(tf.layers.timeDistributed({
   layer: tf.layers.dense({units: 8}),
   inputShape: [10, 16],
}));

// Now model.outputShape = [null, 10, 8].
// The output will then have shape `[32, 10, 8]`.

// In subsequent layers, there is no need for `inputShape`:
model.add(tf.layers.timeDistributed({layer: tf.layers.dense({units: 32})}));
console.log(JSON.stringify(model.outputs[0].shape));
```

--------------------------------

### tf.initializers.randomNormal

Source: https://js.tensorflow.org/api/latest/index

Initializer that generates random values drawn from a normal distribution.

```APIDOC
## tf.initializers.randomNormal (args)

### Description
Initializer that generates random values initialized to a normal distribution.

### Method
Function

### Endpoint
N/A (Client-side JavaScript function)

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
* **args** (Object) - Configuration object for the initializer.
* **mean** (number) - Optional. Mean of the random values to generate. Defaults to 0.
* **stddev** (number) - Optional. Standard deviation of the random values to generate. Defaults to 1.
* **seed** (number) - Optional. Used to seed the random generator.

### Request Example
```json
{
  "args": {
    "mean": 0.5,
    "stddev": 0.2,
    "seed": 104
  }
}
```

### Response
#### Success Response (tf.initializers.Initializer)
Returns an instance of tf.initializers.Initializer.

#### Response Example
```json
{
  "instance": "tf.initializers.Initializer"
}
```
```

--------------------------------

### Calculate Tensor Size from Shape with tf.util.sizeFromShape

Source: https://js.tensorflow.org/api/latest/index

Calculates the total number of elements in a tensor given its shape. This is a utility for understanding tensor dimensions.

```javascript
const shape = [3, 4, 2];
const size = tf.util.sizeFromShape(shape);
console.log(size);
```

--------------------------------

### tf.dilation2d

Source: https://js.tensorflow.org/api/latest/index

Computes the grayscale dilation over the input tensor.

```APIDOC
## POST /tf.dilation2d

### Description
Computes the grayscale dilation over the input `x`.

### Method
POST

### Endpoint
/tf.dilation2d

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **x** (tf.Tensor3D|tf.Tensor4D|TypedArray|Array) - Required - The input tensor, rank 3 or rank 4 of shape `[batch, height, width, depth]`. If rank 3, batch of 1 is assumed.
- **filter** (tf.Tensor3D|TypedArray|Array) - Required - The filter tensor, rank 3, of shape `[filterHeight, filterWidth, depth]`.
- **strides** ([number, number]|number) - Required - The strides of the sliding window for each dimension of the input tensor: `[strideHeight, strideWidth]`. If `strides` is a single number, then `strideHeight == strideWidth`.
- **pad** ('valid'|'same') - Required - The type of padding algorithm.
- **dilations** ([number, number]|number) - Optional - The dilation rates: `[dilationHeight, dilationWidth]` in which we sample input values across the height and width dimensions for atrous morphological dilation. Defaults to `[1, 1]`. If `dilations` is a single number, then `dilationHeight == dilationWidth`. If it is greater than 1, then all values of `strides` must be 1.
- **dataFormat** ('NHWC') - Optional - Specify the data format of the input and output data. Defaults to 'NHWC'. Only 'NHWC' is currently supported.

### Request Example
```json
{
  "x": [ ... ],
  "filter": [ ... ],
  "strides": [ 1, 1 ],
  "pad": "same",
  "dilations": [ 2, 2 ],
  "dataFormat": "NHWC"
}
```

### Response
#### Success Response (200)
- **output** (tf.Tensor3D|tf.Tensor4D) - The output tensor after dilation.

#### Response Example
```json
{
  "output": [ ... ]
}
```
```

--------------------------------

### 3D Max Pooling with TensorFlow.js

Source: https://js.tensorflow.org/api/latest/index

Computes the 3D max pooling operation for a given input tensor, filter size, strides, and padding.

```javascript
const x = tf.tensor5d([1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 2, 2, 1]);
const result = tf.maxPool3d(x, 2, 1, 'valid');
result.print();
```

--------------------------------

### tf.truncatedNormal

Source: https://js.tensorflow.org/api/latest/index

Creates a tf.Tensor with values sampled from a truncated normal distribution.

```APIDOC
## POST /tf.truncatedNormal

### Description
Creates a tf.Tensor with values sampled from a truncated normal distribution. The generated values follow a normal distribution with specified mean and standard deviation, except that values whose magnitude is more than 2 standard deviations from the mean are dropped and re-picked.

### Method
POST

### Endpoint
/tf.truncatedNormal

### Parameters
#### Query Parameters
- **shape** (number[]) - Required - An array of integers defining the output tensor shape.
- **mean** (number) - Optional - The mean of the normal distribution.
- **stdDev** (number) - Optional - The standard deviation of the normal distribution.
- **dtype** ('float32'|'int32') - Optional - The data type of the output tensor.
- **seed** (number) - Optional - The seed for the random number generator.

### Request Example
```json
{
  "shape": [2, 2]
}
```

### Response
#### Success Response (200)
- **tf.Tensor** - A tensor with values sampled from a truncated normal distribution.

#### Response Example
```json
{
  "tensor": [[0.5, -1.2], [0.8, 1.5]]
}
```
```

--------------------------------

### CategoryEncoding Layer in TensorFlow.js

Source: https://js.tensorflow.org/api/latest/index

The CategoryEncoding layer in TensorFlow.js preprocesses integer features by encoding them into a categorical representation. It supports 'oneHot', 'multiHot', and 'count' output modes, allowing for flexible data transformation. Inputs must be integers within the specified 'numTokens' range.

```javascript
const layer = tf.layers.categoryEncoding({
  numTokens: 10,
  outputMode: 'oneHot'
});

const input = tf.tensor1d([0, 2, 4], 'int32');
console.log(layer.apply(input).toString());
// output: 
// Tensor
//    [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
//     [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
//     [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]]
```

--------------------------------

### Apply L1 Regularization in TensorFlow.js

Source: https://js.tensorflow.org/api/latest/index

The tf.regularizers.l1() function applies L1 regularization to penalize large weights by adding a term to the loss function: loss += sum(l1 * abs(x)). It requires an optional configuration object and takes an 'l1' rate.

```javascript
tf.regularizers.l1 (config?)
```

--------------------------------

### tf.spaceToBatchND

Source: https://js.tensorflow.org/api/latest/index

Divides "spatial" dimensions `[1, ..., M]` of the input into a grid of blocks of shape `blockShape`, and interleaves these blocks with the "batch" dimension (0).

```APIDOC
## tf.spaceToBatchND

### Description
Divides "spatial" dimensions `[1, ..., M]` of the input into a grid of blocks of shape `blockShape`, and interleaves these blocks with the "batch" dimension (0) such that in the output, the spatial dimensions `[1, ..., M]` correspond to the position within the grid, and the batch dimension combines both the position within a spatial block and the original batch position. Prior to division into blocks, the spatial dimensions of the input are optionally zero padded according to `paddings`.

### Method
`tf.spaceToBatchND(x, blockShape, paddings)`

### Parameters
#### Parameters
- **x** (tf.Tensor|TypedArray|Array) - A tf.Tensor. N-D with `x.shape` = `[batch] + spatialShape + remainingShape`, where `spatialShape` has `M` dimensions.
- **blockShape** (number[]) - A 1-D array. Must have shape `[M]`, all values must be >= 1.
- **paddings** (number[][]) - A 2-D array of shape `[M, 2]` specifying the padding for spatial dimensions.

### Request Example
```javascript
const x = tf.tensor4d([1, 2, 3, 4], [1, 2, 2, 1]);
const blockShape = [2, 2];
const paddings = [[0, 0], [0, 0]];

x.spaceToBatchND(blockShape, paddings).print();
```

### Response
#### Success Response (200)
- **output** (tf.Tensor) - The reshaped tensor after space to batch transformation.
```

--------------------------------

### Apply L1 and L2 Regularization in TensorFlow.js

Source: https://js.tensorflow.org/api/latest/index

The tf.regularizers.l1l2() function applies both L1 and L2 regularization. It adds a combined penalty to the loss: loss += sum(l1 * abs(x)) + sum(l2 * x^2). It accepts optional 'l1' and 'l2' rates.

```javascript
tf.regularizers.l1l2 (config?)
```

--------------------------------

### Compute Sparse Segment Sum with TensorFlow.js

Source: https://js.tensorflow.org/api/latest/index

Computes the sum along sparse segments of a tensor. This function requires data, indices, and segment IDs. It is optimized for WebGPU, WebGL, WASM, and CPU backends.

```javascript
const c = tf.tensor2d([[1,2,3,4], [-1,-2,-3,-4], [5,6,7,8]]);
// Select two rows, one segment.
const result1 = tf.sparse.sparseSegmentSum(c,
                                           tf.tensor1d([0, 1], 'int32'),
                                           tf.tensor1d([0, 0], 'int32'));
result1.print(); // [[0, 0, 0, 0]]

// Select two rows, two segments.
const result2 = tf.sparse.sparseSegmentSum(c,
                                           tf.tensor1d([0, 1], 'int32'),
                                           tf.tensor1d([0, 1], 'int32'));
result2.print(); // [[1, 2, 3, 4], [-1, -2, -3, -4]]

// Select all rows, two segments.
const result3 = tf.sparse.sparseSegmentSum(c,
                                           tf.tensor1d([0, 1, 2], 'int32'),
                                           tf.tensor1d([0, 0, 1], 'int32'));
result3.print(); // [[0, 0, 0, 0], [5, 6, 7, 8]]
```

--------------------------------

### tf.real

Source: https://js.tensorflow.org/api/latest/index

Returns the real part of a complex (or real) tensor.

```APIDOC
## POST /tf.real

### Description
Returns the real part of a complex (or real) tensor. Given a tensor input, this operation returns a tensor of type float that is the real part of each element in input considered as a complex number. If the input is real, it simply makes a clone.

### Method
POST

### Endpoint
/tf.real

### Parameters
#### Query Parameters
- **input** (tf.Tensor|TypedArray|Array) - Required - The input tensor.

### Request Example
```json
{
  "input": {"real": [-2.25, 3.25], "imag": [4.75, 5.75]}
}
```

### Response
#### Success Response (200)
- **tf.Tensor** - A tensor containing the real part of the input.

#### Response Example
```json
{
  "tensor": [-2.25, 3.25]
}
```
```

--------------------------------

### Crop 2D Image Tensor with TensorFlow.js

Source: https://js.tensorflow.org/api/latest/index

The Cropping2D layer crops an input image tensor from the top, bottom, left, and right sides. It supports different data formats and input shapes. The cropping can be applied symmetrically or asymmetrically.

```javascript
const model = tf.sequential();
model.add(tf.layers.cropping2D({
  cropping: [[2, 2], [2, 2]],
  inputShape: [128, 128, 3]
}));
// Output shape is now [batch, 124, 124, 3]
```

--------------------------------

### Repeat Dataset Multiple Times with JavaScript

Source: https://js.tensorflow.org/api/latest/index

The `repeat` method creates a new dataset that contains elements from the original dataset repeated a specified number of times. If the count is not provided or is negative, the dataset is repeated indefinitely. Note that repetitions might produce different elements if the dataset depends on global state like a random number generator.

```javascript
const a = tf.data.array([1, 2, 3]).repeat(3);
await a.forEachAsync(e => console.log(e));

```

--------------------------------

### Create a Transposed Convolutional Layer (Conv2D)

Source: https://js.tensorflow.org/api/latest/index

This code snippet demonstrates how to create a transposed convolutional layer using tf.layers.conv2dTranspose. This layer is useful for upsampling feature maps in deep learning models. It requires specifying the number of filters, kernel size, and strides, with optional parameters for padding, activation, and data format.

```javascript
const tf = require('@tensorflow/tfjs');

// Example: Create a transposed convolutional layer
const conv2dTransposeLayer = tf.layers.conv2dTranspose({
  filters: 32,
  kernelSize: 3,
  strides: 2,
  padding: 'same',
  activation: 'relu',
  inputShape: [28, 28, 1] // Example input shape for a 28x28 grayscale image
});

console.log(conv2dTransposeLayer.getConfig());
```

--------------------------------

### tf.topk: Find k largest elements and their indices (JavaScript)

Source: https://js.tensorflow.org/api/latest/index

The tf.topk operation identifies the `k` largest values and their corresponding indices along the last dimension of a tensor. If `sorted` is true, the output values will be in descending order. If elements are equal, the one with the lower index appears first.

```javascript
const a = tf.tensor2d([[1, 5], [4, 3]]);
const {values, indices} = tf.topk(a);
values.print();
indices.print();
```

--------------------------------

### tf.lowerBound: Find insertion points in sorted sequences (JavaScript)

Source: https://js.tensorflow.org/api/latest/index

The tf.lowerBound operation finds the indices where elements should be inserted into a sorted tensor to maintain order. It operates on the innermost axis and assumes the input sequence is sorted along this axis. If the sequence is not sorted, the output is undefined. This is equivalent to tf.searchSorted with the 'left' option.

```javascript
const edges = tf.tensor1d([-1, 3.3, 9.1, 10.0]);
let values = tf.tensor1d([0.0, 4.1, 12.0]);
const result1 = tf.lowerBound(edges, values);
result1.print(); // [1, 2, 4]

const seq = tf.tensor1d([0, 3, 9, 10, 10]);
values = tf.tensor1d([0, 4, 10]);
const result2 = tf.lowerBound(seq, values);
result2.print(); // [0, 2, 3]

const sortedSequence = tf.tensor2d([[0., 3., 8., 9., 10.],
                                     [1., 2., 3., 4., 5.]]);
values = tf.tensor2d([[9.8, 2.1, 4.3],
                       [0.1, 6.6, 4.5, ]]);
const result3 = tf.lowerBound(sortedSequence, values);
result3.print(); // [[4, 1, 2], [0, 5, 4]]
```

--------------------------------

### 1D Convolution - JavaScript

Source: https://js.tensorflow.org/api/latest/index

Computes a 1D convolution over the input tensor. This operation is supported on WebGPU, WebGL, Wasm, and CPU. It takes the input tensor, a filter tensor, stride, padding type, and optional data format, dilation, and dimension rounding mode.

```javascript
tf.conv1d(x, filter, stride, pad, dataFormat?, dilation?, dimRoundingMode?)
```

--------------------------------

### Apply L2 Regularization in TensorFlow.js

Source: https://js.tensorflow.org/api/latest/index

The tf.regularizers.l2() function applies L2 regularization to penalize large weights by adding a term to the loss function: loss += sum(l2 * x^2). It requires an optional configuration object and takes an 'l2' rate.

```javascript
tf.regularizers.l2 (config?)
```

--------------------------------

### Compute Recall with TensorFlow.js

Source: https://js.tensorflow.org/api/latest/index

Computes the recall of predictions against true labels. It's used to measure the proportion of actual positive instances that were correctly identified. Input Tensors (yTrue, yPred) are expected to contain only 0-1 values. The function supports webgpu, webgl, wasm, and cpu execution.

```javascript
const x = tf.tensor2d(
    [
      [0, 0, 0, 1],
      [0, 1, 0, 0],
      [0, 0, 0, 1],
      [1, 0, 0, 0],
      [0, 0, 1, 0]
    ]
);

const y = tf.tensor2d(
    [
      [0, 0, 1, 0],
      [0, 1, 0, 0],
      [0, 0, 0, 1],
      [0, 1, 0, 0],
      [0, 1, 0, 0]
    ]
);

const recall = tf.metrics.recall(x, y);
recall.print();
```

--------------------------------

### tf.lessEqual (a, b)

Source: https://js.tensorflow.org/api/latest/index

Returns the truth value of (a <= b) element-wise. Supports broadcasting.

```APIDOC
## POST /tf.lessEqual

### Description
Returns the truth value of (a <= b) element-wise. Supports broadcasting.

### Method
POST

### Endpoint
/tf.lessEqual

### Parameters
#### Request Body
- **a** (tf.Tensor|TypedArray|Array) - Required - The first input tensor.
- **b** (tf.Tensor|TypedArray|Array) - Required - The second input tensor. Must have the same dtype as `a`.

### Request Example
```json
{
  "a": [1, 2, 3],
  "b": [2, 2, 2]
}
```

### Response
#### Success Response (200)
- **result** (tf.Tensor) - The resulting tensor with truth values.

#### Response Example
```json
{
  "result": [true, true, true]
}
```
```

--------------------------------

### tf.cumprod: Compute Cumulative Product of a Tensor

Source: https://js.tensorflow.org/api/latest/index

Computes the cumulative product of a TensorFlow.js tensor along a specified axis. It supports exclusive and reverse cumulative product calculations. Supports multiple backends like webgpu, webgl, wasm, and cpu.

```javascript
const x = tf.tensor([1, 2, 3, 4]);
x.cumprod().print();
```

```javascript
const x = tf.tensor([[1, 2], [3, 4]]);
x.cumprod().print();
```

--------------------------------

### Multiply Tensors Element-wise with TensorFlow.js

Source: https://js.tensorflow.org/api/latest/index

Multiplies two tf.Tensors element-wise (A * B), supporting broadcasting. Use either `a.mul(b)` or `tf.mul(a, b)`. A strict version, `tf.mulStrict`, is available for asserting identical shapes.

```javascript
const a = tf.tensor1d([1, 2, 3, 4]);
const b = tf.tensor1d([2, 3, 4, 5]);

a.mul(b).print();  // or tf.mul(a, b)
```

```javascript
// Broadcast mul a with b.
const a = tf.tensor1d([1, 2, 3, 4]);
const b = tf.scalar(5);

a.mul(b).print();  // or tf.mul(a, b)
```

--------------------------------

### Compute Hyperbolic Tangent of Tensor Elements (JavaScript)

Source: https://js.tensorflow.org/api/latest/index

Computes the element-wise hyperbolic tangent of a tensor. Supports CPU, WebGL, WebGPU, and WASM backends. Takes a tensor, TypedArray, or Array as input and returns a tensor.

```javascript
const x = tf.tensor1d([0, 1, -1, .7]);

x.tanh().print();  // or tf.tanh(x)
```

--------------------------------

### Conv2D Transpose Layer with Data Format Specification

Source: https://js.tensorflow.org/api/latest/index

This snippet illustrates how to specify the data format for the tf.layers.conv2dTranspose layer. You can choose between 'channelsFirst' (input shape: [batch, channels, rows, cols]) or 'channelsLast' (input shape: [batch, rows, cols, channels]). 'channelsLast' is the default format.

```javascript
const tf = require('@tensorflow/tfjs');

// Example: Conv2D Transpose with 'channelsFirst' data format
const conv2dTransposeChannelsFirst = tf.layers.conv2dTranspose({
  filters: 16,
  kernelSize: 3,
  strides: 1,
  padding: 'same',
  dataFormat: 'channelsFirst',
  inputShape: [3, 32, 32] // Example input shape for channelsFirst
});

console.log(conv2dTransposeChannelsFirst.getConfig());

// Example: Conv2D Transpose with 'channelsLast' data format (default)
const conv2dTransposeChannelsLast = tf.layers.conv2dTranspose({
  filters: 16,
  kernelSize: 3,
  strides: 1,
  padding: 'same',
  dataFormat: 'channelsLast',
  inputShape: [32, 32, 3] // Example input shape for channelsLast
});

console.log(conv2dTransposeChannelsLast.getConfig());
```

--------------------------------

### Calculate Binary Crossentropy - JavaScript

Source: https://js.tensorflow.org/api/latest/index

Computes the binary crossentropy between two tensors. This metric function is suitable for binary classification problems and supports multiple backends.

```javascript
const x = tf.tensor2d([[0], [1], [1], [1]]);
const y = tf.tensor2d([[0], [0], [0.5], [1]]);
const crossentropy = tf.metrics.binaryCrossentropy(x, y);
crossentropy.print();
```

--------------------------------

### Masking Layer

Source: https://js.tensorflow.org/api/latest/index

Masks a sequence by using a mask value to skip timesteps.

```APIDOC
## tf.layers.masking

### Description
Masks a sequence by using a mask value to skip timesteps. If all features for a given sample timestep are equal to `mask_value`, then the sample timestep will be masked (skipped) in all downstream layers (as long as they support masking). If any downstream layer does not support masking yet receives such an input mask, an exception will be raised.

### Method
Not applicable (Layer constructor)

### Endpoint
Not applicable (Layer constructor)

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
None

##### Parameters for Layer Construction
- **args** (Object) - Optional
- **maskValue** (number) - Masking Value. Defaults to `0.0`.
- **inputShape** (Array<number | null>) - If defined, will be used to create an input layer to insert before this layer. If both `inputShape` and `batchInputShape` are defined, `batchInputShape` will be used. This argument is only applicable to input layers (the first layer of a model).
- **batchInputShape** (Array<number | null>) - If defined, will be used to create an input layer to insert before this layer. If both `inputShape` and `batchInputShape` are defined, `batchInputShape` will be used. This argument is only applicable to input layers (the first layer of a model).
- **batchSize** (number) - If `inputShape` is specified and `batchInputShape` is _not_ specified, `batchSize` is used to construct the `batchInputShape`: `[batchSize, ...inputShape]`
- **dtype** ('float32'|'int32'|'bool'|'complex64'|'string') - The data-type for this layer. Defaults to 'float32'. This argument is only applicable to input layers (the first layer of a model).
- **name** (string) - Name for this layer.
- **trainable** (boolean) - Whether the weights of this layer are updatable by `fit`. Defaults to true.
- **weights** (Array<tf.Tensor>) - Initial weight values of the layer.
- **inputDType** ('float32'|'int32'|'bool'|'complex64'|'string') - Legacy support. Do not use for new code.

### Request Example
```json
{
  "example": "// TensorFlow.js code for creating Masking layer\nconst layer = tf.layers.masking({maskValue: -1});"
}
```

### Response
#### Success Response (Layer Object)
- **Layer Object** (object) - A TensorFlow.js layer object configured for masking.

#### Response Example
```json
{
  "example": "// Masking layer object"
}
```
```

--------------------------------

### tf.losses.logLoss

Source: https://js.tensorflow.org/api/latest/index

Computes the log loss between two tensors. Also known as cross-entropy loss, it's widely used for binary and multi-class classification problems.

```APIDOC
## POST tf.losses.logLoss

### Description
Computes the log loss between two tensors. Also known as cross-entropy loss, it's widely used for binary and multi-class classification problems.

### Method
POST

### Endpoint
tf.losses.logLoss

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **labels** (tf.Tensor|TypedArray|Array) - Required - The ground truth output tensor, same dimensions as 'predictions'.
- **predictions** (tf.Tensor|TypedArray|Array) - Required - The predicted outputs.
- **weights** (tf.Tensor|TypedArray|Array) - Optional - Tensor whose rank is either 0, or the same rank as `labels`, and must be broadcastable to `labels` (i.e., all dimensions must be either `1`, or the same as the corresponding `losses` dimension).
- **epsilon** (number) - Optional - A small increment to avoid taking log of zero.
- **reduction** (Reduction) - Optional - Type of reduction to apply to loss. Should be of type `Reduction`.

### Request Example
```json
{
  "labels": "tensor_data",
  "predictions": "tensor_data",
  "weights": "tensor_data",
  "epsilon": 1e-7,
  "reduction": "MEAN"
}
```

### Response
#### Success Response (200)
- **tensor** (tf.Tensor) - The computed log loss.

#### Response Example
```json
{
  "result": "tensor_data"
}
```
```

--------------------------------

### tf.range

Source: https://js.tensorflow.org/api/latest/index

Creates a new tf.Tensor1D filled with numbers in a specified range.

```APIDOC
## POST /tf.range

### Description
Creates a new tf.Tensor1D filled with the numbers in the range provided. The tensor is a half-open interval meaning it includes start, but excludes stop. Decrementing ranges and negative step values are also supported.

### Method
POST

### Endpoint
/tf.range

### Parameters
#### Query Parameters
- **start** (number) - Required - An integer start value
- **stop** (number) - Required - An integer stop value
- **step** (number) - Optional - An integer increment (will default to 1 or -1)
- **dtype** ('float32'|'int32') - Optional - The data type of the output tensor. Defaults to 'float32'.

### Request Example
```json
{
  "start": 0,
  "stop": 9,
  "step": 2
}
```

### Response
#### Success Response (200)
- **tf.Tensor1D** - The created tensor.

#### Response Example
```json
{
  "tensor": [0, 2, 4, 6, 8]
}
```
```

--------------------------------

### MinMaxNorm Constraint for TensorFlow.js Layers

Source: https://js.tensorflow.org/api/latest/index

Applies a MinMaxNorm constraint to layer weights, enforcing a range for the norm of incoming weights. It includes a rate parameter to control the enforcement intensity.

```javascript
const constraint = tf.constraints.minMaxNorm ({minValue: 0.5, maxValue: 1.0, axis: 0, rate: 1.0});
```

--------------------------------

### tf.image.nonMaxSuppression

Source: https://js.tensorflow.org/api/latest/index

Performs non-maximum suppression (NMS) on bounding boxes based on their intersection over union (IOU) scores. This is crucial for object detection to eliminate redundant overlapping boxes.

```APIDOC
## tf.image.nonMaxSuppression

### Description
Performs non maximum suppression of bounding boxes based on iou (intersection over union).

### Method
N/A (Function)

### Endpoint
N/A (Function)

### Parameters
#### Path Parameters
N/A

#### Query Parameters
N/A

#### Request Body
*   **boxes** (tf.Tensor2D|TypedArray|Array) - Required - A 2D tensor of shape `[numBoxes, 4]`. Each entry is `[y1, x1, y2, x2]`, where `(y1, x1)` and `(y2, x2)` are the corners of the bounding box.
*   **scores** (tf.Tensor1D|TypedArray|Array) - Required - A 1D tensor providing the box scores of shape `[numBoxes]`.
*   **maxOutputSize** (number) - Required - The maximum number of boxes to be selected.
*   **iouThreshold** (number) - Optional - A float representing the threshold for deciding whether boxes overlap too much with respect to IOU. Must be between [0, 1]. Defaults to 0.5 (50% box overlap).
*   **scoreThreshold** (number) - Optional - A threshold for deciding when to remove boxes based on score. Defaults to -inf, which means any score is accepted.

### Request Example
```json
{
  "boxes": "tf.Tensor2D",
  "scores": "tf.Tensor1D",
  "maxOutputSize": 10,
  "iouThreshold": 0.5,
  "scoreThreshold": 0.1
}
```

### Response
#### Success Response (200)
*   **tf.Tensor1D** - A 1D tensor containing the indices of the selected boxes.

#### Response Example
```json
{
  "selectedIndices": "tf.Tensor1D"
}
```
```

--------------------------------

### Compute Tensor Power Element-wise with tf.pow

Source: https://js.tensorflow.org/api/latest/index

Computes the power of one tf.Tensor to another element-wise, supporting broadcasting. The result's dtype is the upcasted type of the base and exponent dtypes. Optimized for webgpu, webgl, wasm, and cpu backends.

```javascript
const a = tf.tensor([[2, 3], [4, 5]])
const b = tf.tensor([[1, 2], [3, 0]]).toInt();

a.pow(b).print();  // or tf.pow(a, b)
```

```javascript
const a = tf.tensor([[1, 2], [3, 4]])
const b = tf.tensor(2).toInt();

a.pow(b).print();  // or tf.pow(a, b)
```

--------------------------------

### tf.scalar

Source: https://js.tensorflow.org/api/latest/index

Creates a rank-0 tf.Tensor (scalar) with the provided value and dtype.

```APIDOC
## POST /api/v1/scalar

### Description
Creates a rank-0 tf.Tensor (scalar) with the provided value and dtype.
The same functionality can be achieved with tf.tensor(), but in general we recommend using tf.scalar() as it makes the code more readable.

### Method
POST

### Endpoint
/api/v1/scalar

### Parameters
#### Request Body
- **value** (number|boolean|string|Uint8Array) - Required - The value of the scalar.
- **dtype** ('float32'|'int32'|'bool'|'complex64'|'string') - Optional - The data type. Optional 

### Response
#### Success Response (200)
- **scalar** (tf.Scalar) - The created scalar tensor.

#### Response Example
```json
{
  "scalar": "<Scalar Tensor Object>"
}
```
```

--------------------------------

### AveragePooling2D API

Source: https://js.tensorflow.org/api/latest/index

Performs average pooling operation for spatial 2D data. It downscales the input by a specified pool size.

```APIDOC
## AveragePooling2D

### Description
Average pooling operation for spatial data. It downscales the input by halving or specifying the input dimensions.

### Method
`tf.layers.averagePooling2d`

### Endpoint
N/A (This is a layer function, not a REST endpoint)

### Parameters
#### Arguments Object (args)
- **poolSize** (number | [number, number]) - Required - Factors by which to downscale in each dimension [vertical, horizontal].
- **strides** (number | [number, number] | null) - Optional - The size of the stride in each dimension. Defaults to `poolSize`.
- **padding** ('valid' | 'same' | 'causal') - Optional - The padding type to use.
- **dataFormat** ('channelsFirst' | 'channelsLast') - Optional - The data format to use.
- **inputShape** (Array<number | null>) - Optional - Used to create an input layer if not already defined.
- **batchInputShape** (Array<number | null>) - Optional - Used to create an input layer if not already defined.
- **batchSize** (number) - Optional - Used to construct `batchInputShape` if `inputShape` is provided.
- **dtype** ('float32' | 'int32' | 'bool' | 'complex64' | 'string') - Optional - The data-type for this layer. Defaults to 'float32'.
- **name** (string) - Optional - Name for this layer.
- **trainable** (boolean) - Optional - Whether the weights of this layer are updatable. Defaults to true.
- **weights** (Array<tf.Tensor>) - Optional - Initial weight values of the layer.
- **inputDType** ('float32' | 'int32' | 'bool' | 'complex64' | 'string') - Optional - Legacy support, do not use for new code.

### Request Example
```json
{
  "poolSize": [2, 2],
  "strides": [1, 1],
  "padding": "valid",
  "dataFormat": "channelsLast"
}
```

### Response
#### Success Response (Output Shape)
- **Output Shape** (4D tensor) - Shape depends on `dataFormat` and pooling parameters.
  - If `dataFormat === CHANNEL_LAST`: `[batchSize, pooledRows, pooledCols, channels]`
  - If `dataFormat === CHANNEL_FIRST`: `[batchSize, channels, pooledRows, pooledCols]`

#### Response Example
(Output shape depends on input and parameters, not a fixed JSON structure)
```

--------------------------------

### Assert Condition with TensorFlow.js Utility Function

Source: https://js.tensorflow.org/api/latest/index

The `tf.util.assert` function verifies if a given boolean expression is true. If the expression is false, it throws an error with a provided message. The message is supplied as a function for performance optimization.

```javascript
const x = 2;
tf.util.assert(x === 2, 'x is not 2');

```

--------------------------------

### tf.layers.globalMaxPooling1D

Source: https://js.tensorflow.org/api/latest/index

Global max pooling operation for temporal data. It reduces the input tensor by taking the maximum value across the time steps.

```APIDOC
## tf.layers.globalMaxPooling1D

### Description
Global max pooling operation for temporal data. Reduces the input tensor by taking the maximum value across the time steps.

### Method
*Construction*

### Endpoint
`tf.layers.globalMaxPooling1D(args)`

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
* **args** (Object) Optional
  * **inputShape** ((null | number)[]) If defined, will be used to create an input layer to insert before this layer. If both `inputShape` and `batchInputShape` are defined, `batchInputShape` will be used. This argument is only applicable to input layers (the first layer of a model).
  * **batchInputShape** ((null | number)[]) If defined, will be used to create an input layer to insert before this layer. If both `inputShape` and `batchInputShape` are defined, `batchInputShape` will be used. This argument is only applicable to input layers (the first layer of a model).
  * **batchSize** (number) If `inputShape` is specified and `batchInputShape` is _not_ specified, `batchSize` is used to construct the `batchInputShape`: `[batchSize, ...inputShape]`
  * **dtype** ('float32'|'int32'|'bool'|'complex64'|'string') The data-type for this layer. Defaults to 'float32'. This argument is only applicable to input layers (the first layer of a model).
  * **name** (string) Name for this layer.
  * **trainable** (boolean) Whether the weights of this layer are updatable by `fit`. Defaults to true.
  * **weights** (tf.Tensor[]) Initial weight values of the layer.
  * **inputDType** ('float32'|'int32'|'bool'|'complex64'|'string') Legacy support. Do not use for new code.

### Request Example
```json
{
  "args": {
    "poolSize": 2,
    "strides": 1,
    "padding": "valid"
  }
}
```

### Response
#### Success Response (200)
Returns: GlobalMaxPooling1D layer instance

#### Response Example
```json
{
  "layerType": "GlobalMaxPooling1D"
}
```
```

--------------------------------

### Matrix Multiplication with TensorFlow.js

Source: https://js.tensorflow.org/api/latest/index

Computes the dot product of two matrices (A * B) using the matMul function. This operation requires both inputs to be matrices. Optional boolean parameters transposeA and transposeB can be used to transpose the matrices before multiplication. Supported on webgpu, webgl, wasm, and cpu.

```javascript
const a = tf.tensor2d([1, 2], [1, 2]);
const b = tf.tensor2d([1, 2, 3, 4], [2, 2]);

a.matMul(b).print();  // or tf.matMul(a, b)
```

--------------------------------

### Find Indices of Minimum Values in Tensor

Source: https://js.tensorflow.org/api/latest/index

Returns the indices of the minimum values along a specified axis in a tensor. The resulting tensor has the same shape as the input tensor with the dimension along the specified axis removed. Supports webgpu, webgl, wasm, and cpu backends.

```javascript
const x = tf.tensor1d([1, 2, 3]);

x.argMin().print();  // or tf.argMin(x)
```

```javascript
const x = tf.tensor2d([1, 2, 4, 3], [2, 2]);

const axis = 1;
x.argMin(axis).print();  // or tf.argMin(x, axis)
```

--------------------------------

### LeakyReLU Activation Layer

Source: https://js.tensorflow.org/api/latest/index

The LeakyReLU layer allows a small gradient when the unit is not active. It follows `f(x) = alpha * x for x < 0.` and `f(x) = x for x >= 0.`

```APIDOC
## LeakyReLU Activation Layer

### Description
Allows a small gradient when the unit is not active. It follows `f(x) = alpha * x for x < 0.` and `f(x) = x for x >= 0.`

### Method
`tf.layers.leakyReLU`

### Endpoint
N/A (Layer definition)

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **args** (Object) Optional
- **alpha** (number) - Optional - Float `>= 0`. Negative slope coefficient. Defaults to `0.3`.
- **inputShape** ((null | number)[]) - Optional - If defined, will be used to create an input layer to insert before this layer. If both `inputShape` and `batchInputShape` are defined, `batchInputShape` will be used. This argument is only applicable to input layers (the first layer of a model).
- **batchInputShape** ((null | number)[]) - Optional - If defined, will be used to create an input layer to insert before this layer. If both `inputShape` and `batchInputShape` are defined, `batchInputShape` will be used. This argument is only applicable to input layers (the first layer of a model).
- **batchSize** (number) - Optional - If `inputShape` is specified and `batchInputShape` is _not_ specified, `batchSize` is used to construct the `batchInputShape`: `[batchSize, ...inputShape]`
- **dtype** ('float32'|'int32'|'bool'|'complex64'|'string') - Optional - The data-type for this layer. Defaults to 'float32'. This argument is only applicable to input layers (the first layer of a model).
- **name** (string) - Optional - Name for this layer.
- **trainable** (boolean) - Optional - Whether the weights of this layer are updatable by `fit`. Defaults to true.
- **weights** (tf.Tensor[]) - Optional - Initial weight values of the layer.
- **inputDType** ('float32'|'int32'|'bool'|'complex64'|'string') - Optional - Legacy support. Do not use for new code.

### Request Example
```json
{
  "alpha": 0.5
}
```

### Response
#### Success Response (200)
- **Layer Output** (Tensor) - Same shape as the input.

#### Response Example
```json
// Tensor output representing the layer's activation
```
```

--------------------------------

### Find Maximum Element-wise with tf.maximum

Source: https://js.tensorflow.org/api/latest/index

Returns the maximum of two tf.Tensors element-wise (a > b ? a : b). Supports broadcasting and can be called on a tensor or as a standalone function. Optimized for webgpu, webgl, wasm, and cpu backends.

```javascript
const a = tf.tensor1d([1, 4, 3, 16]);
const b = tf.tensor1d([1, 2, 9, 4]);

a.maximum(b).print();  // or tf.maximum(a, b)
```

```javascript
// Broadcast maximum a with b.
const a = tf.tensor1d([2, 4, 6, 8]);
const b = tf.scalar(5);

a.maximum(b).print();  // or tf.maximum(a, b)
```

--------------------------------

### tf.transpose

Source: https://js.tensorflow.org/api/latest/index

Transposes the tf.Tensor. Permutes the dimensions according to `perm`.

```APIDOC
## transpose / tf.transpose

### Description
Transposes the tf.Tensor. Permutes the dimensions according to `perm`. The returned tf.Tensor's dimension `i` will correspond to the input dimension `perm[i]`. If `perm` is not given, it is set to `[n-1...0]`, where `n` is the rank of the input tf.Tensor. Hence by default, this operation performs a regular matrix transpose on 2-D input tf.Tensors.

### Method
Various (e.g., `x.transpose()` or `tf.transpose(x)`)

### Endpoint
N/A (function call)

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **x** (tf.Tensor) - The input tensor to transpose.
- **perm** (number[]) - Optional. A permutation of the dimensions of the input tensor.
- **conjugate** (boolean) - Optional. If true, the complex conjugate is taken before transposing.

### Request Example
```json
{
  "example": "const matrix = tf.tensor2d([[1, 2], [3, 4]]);\nmatrix.transpose().print(); // [[1, 3], [2, 4]]"
}
```

### Response
#### Success Response (200)
- **result** (tf.Tensor) - The transposed tensor.

#### Response Example
```json
{
  "example": "Output tensor representing the transposed input"
}
```
```

--------------------------------

### Local Response Normalization

Source: https://js.tensorflow.org/api/latest/index

Normalizes activations within a local neighborhood across or within channels. This function is typically used for convolutional neural networks.

```javascript
const x = tf.tensor4d([[[[1], [2]], [[3], [4]]]], [1, 2, 2, 1]);
tf.localResponseNormalization(x, 1, 0.1, 0.5, 1.0).print();
```

--------------------------------

### Extract Real Part of Complex Tensor

Source: https://js.tensorflow.org/api/latest/index

Returns the real part of a complex or real tensor. If the input is real, a clone is returned. The output tensor is always of float type. This operation is supported on webgpu, webgl, wasm, and cpu.

```javascript
const x = tf.complex([-2.25, 3.25], [4.75, 5.75]);
tf.real(x).print();
```

--------------------------------

### tf.maxPool3d

Source: https://js.tensorflow.org/api/latest/index

Computes the 3D max pooling operation.

```APIDOC
## POST /tf.maxPool3d

### Description
Computes the 3D max pooling.

### Method
POST

### Endpoint
/tf.maxPool3d

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **x** (tf.Tensor4D|tf.Tensor5D|TypedArray|Array) - Required - The input tensor, of rank 5 or rank 4 of shape `[batch, depth, height, width, inChannels]`.
- **filterSize** ([number, number, number]|number) - Required - The size of the window over which to take the max. If a single number, then the window size is the same in all dimensions.
- **strides** ([number, number, number]|number) - Required - The strides of the convolution: `[strideDepth, strideHeight, strideWidth]`. If strides is a single number, then `strideDepth == strideHeight == strideWidth`.
- **pad** ('valid'|'same') - Required - The type of padding algorithm used in the non-transpose version of the op.
- **dimRoundingMode** ('floor'|'round'|'ceil') - Optional - A string from: 'ceil', 'round', 'floor'. If none is provided, it will default to truncate.
- **dataFormat** ('NHWC'|'NCHW') - Optional - An optional string from: "NHWC", "NCHW". Defaults to "NHWC". Specify the data format of the input and output data. With the default format "NHWC", the data is stored in the order of: [batch, depth, height, width, channels]. Only "NHWC" is currently supported.

### Request Example
```json
{
  "x": [ ... ],
  "filterSize": [ 2, 2, 2 ],
  "strides": [ 1, 1, 1 ],
  "pad": "valid",
  "dimRoundingMode": "floor",
  "dataFormat": "NHWC"
}
```

### Response
#### Success Response (200)
- **output** (tf.Tensor4D|tf.Tensor5D) - The output tensor of the max pooling operation.

#### Response Example
```json
{
  "output": [ ... ]
}
```
```

--------------------------------

### tf.split

Source: https://js.tensorflow.org/api/latest/index

Splits a tf.Tensor into sub tensors along a specified dimension.

```APIDOC
## tf.split

### Description
Splits a tf.Tensor into sub tensors. If `numOrSizeSplits` is a number, splits `x` along dimension `axis` into `numOrSizeSplits` smaller tensors, requiring that `numOrSizeSplits` evenly divides `x.shape[axis]`. If `numOrSizeSplits` is a number array, splits `x` into `numOrSizeSplits.length` pieces.

### Method
`split(x: tf.Tensor | tf.TypedArray | tf.TensorLike, numOrSizeSplits: number | number[], axis?: number): tf.Tensor[]`

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
None

### Request Example
```javascript
const x = tf.tensor2d([1, 2, 3, 4, 5, 6, 7, 8], [2, 4]);
const [a, b] = tf.split(x, 2, 1);
a.print();
b.print();

const [c, d, e] = tf.split(x, [1, 2, 1], 1);
c.print();
d.print();
e.print();
```

### Response
#### Success Response (200)
- **tf.Tensor[]** - An array of tensors.

#### Response Example
```json
{
  "example": "array of tensor data"
}
```
```

--------------------------------

### Register Custom Layer Class with Specified Name in JavaScript

Source: https://js.tensorflow.org/api/latest/index

Shows how to register a custom Layer class with only a specified name ('MyLayer') in TensorFlow.js serialization, while letting the package name default. This is useful when you need to provide a specific identifier for the class.

```javascript
class MyCustomLayer extends tf.layers.Layer {
   static className = 'MyCustomLayer';

   constructor(config) {
     super(config);
   }
}
tf.serialization.registerClass(MyCustomLayer, undefined, "MyLayer");
console.log(tf.serialization.GLOBALCUSTOMOBJECT.get("Custom>MyLayer"));
console.log(tf.serialization.GLOBALCUSTOMNAMES.get(MyCustomLayer));
```

--------------------------------

### Select elements based on a condition

Source: https://js.tensorflow.org/api/latest/index

This function selects elements from tensor `a` where the `condition` is true, and from tensor `b` where the `condition` is false. The `condition` tensor must be of boolean dtype. Broadcasting rules apply to `a` and `b` based on the `condition`'s rank and size.

```javascript
const cond = tf.tensor1d([false, false, true], 'bool');
const a = tf.tensor1d([1 , 2, 3]);
const b = tf.tensor1d([-1, -2, -3]);

a.where(cond, b).print();
```

--------------------------------

### TensorFlow.js: Random Tensor from User Function

Source: https://js.tensorflow.org/api/latest/index

Generates a tensor with values sampled from a user-defined random number generator function using `tf.rand`. This provides flexibility in creating tensors with custom random distributions.

```javascript
tf.rand([2, 2], () => Math.random()).print();
```

--------------------------------

### Create Maximum Layer in JavaScript

Source: https://js.tensorflow.org/api/latest/index

This JavaScript code demonstrates how to create and use the `tf.layers.maximum` layer. This layer computes the element-wise maximum of an array of input tensors. It requires inputs of the same shape and returns a single tensor of that shape. The batch dimension is determined at runtime.

```javascript
const input1 = tf.input({shape: [2, 2]});
const input2 = tf.input({shape: [2, 2]});
const maxLayer = tf.layers.maximum();
const max = maxLayer.apply([input1, input2]);
console.log(JSON.stringify(max.shape));
// Output: [null, 2, 2]
```

--------------------------------

### Decode Bytes to String with tf.decodeString

Source: https://js.tensorflow.org/api/latest/index

Decodes a Uint8Array into a string using a specified encoding. Defaults to 'utf-8' if no encoding is provided. Handles potential errors during decoding.

```javascript
const bytes = new Uint8Array([116, 101, 110, 115, 111, 114, 47, 102, 108, 111, 119]);
const decodedString = tf.decodeString(bytes, 'utf-8');
console.log(decodedString);
```

--------------------------------

### Floor Divide Tensors Element-wise with tf.floorDiv

Source: https://js.tensorflow.org/api/latest/index

Divides two tf.Tensors element-wise and rounds the result down using the floor function. Supports broadcasting and can be called on a tensor or as a standalone function. Optimized for webgpu, webgl, wasm, and cpu backends.

```javascript
const a = tf.tensor1d([1, 4, 9, 16]);
const b = tf.tensor1d([1, 2, 3, 4]);

a.floorDiv(b).print();  // or tf.div(a, b)
```

```javascript
// Broadcast div a with b.
const a = tf.tensor1d([2, 4, 6, 8]);
const b = tf.scalar(2);

a.floorDiv(b).print();  // or tf.floorDiv(a, b)
```

--------------------------------

### tf.less (a, b)

Source: https://js.tensorflow.org/api/latest/index

Returns the truth value of (a < b) element-wise. Supports broadcasting.

```APIDOC
## POST /tf.less

### Description
Returns the truth value of (a < b) element-wise. Supports broadcasting.

### Method
POST

### Endpoint
/tf.less

### Parameters
#### Request Body
- **a** (tf.Tensor|TypedArray|Array) - Required - The first input tensor.
- **b** (tf.Tensor|TypedArray|Array) - Required - The second input tensor. Must have the same dtype as `a`.

### Request Example
```json
{
  "a": [1, 2, 3],
  "b": [2, 2, 2]
}
```

### Response
#### Success Response (200)
- **result** (tf.Tensor) - The resulting tensor with truth values.

#### Response Example
```json
{
  "result": [true, false, false]
}
```
```