import {
  matrix,
  add,
  multiply,
  subtract,
  Matrix,
  transpose,
  zeros,
} from "mathjs";
import { xorData } from "./data/XoR.data";
import { activation_sigmoid, derivative_sigmoid } from "./utils/sigmoid";

class Layer {
  weights: Matrix;
  bias: Matrix;

  neuronsNumber: number;
  parent: Layer | null;
  children: Layer | null;

  lastArguments: Matrix;
  lastInput: Matrix;

  prevDeltaWeights: Matrix | null;

  constructor(neuronsNumber: number, parent?: Layer) {
    this.neuronsNumber = neuronsNumber;
    this.children = null;
    this.prevDeltaWeights = null;

    if (parent) {
      this.parent = parent;
      this.parent.children = this;

      this.weights = zeros(neuronsNumber, parent.neuronsNumber).map(
        () => 4 * Math.random() - 2
      ) as Matrix;
      this.bias = zeros(neuronsNumber, 1).map(
        () => 4 * Math.random() - 2
      ) as Matrix;
    } else {
      this.parent = null;
    }
  }

  calculate(inputData: Matrix) {
    const argumentsData = add(multiply(this.weights, inputData), this.bias);
    const outputData = argumentsData.map(activation_sigmoid);

    this.lastArguments = argumentsData;
    this.lastInput = inputData;

    return outputData;
  }

  train(delta: Matrix) {
    if (!this.parent) {
      return;
    }
    const derivativeData = this.lastArguments.map(derivative_sigmoid);

    if (typeof delta === "number") {
      delta = matrix([delta]);
    }

    const deltaArg = multiply(
      this?.children?.weights ? transpose(this.children.weights) : 1,
      delta
    ).map((elem, pos) => {
      return elem * derivativeData.get(pos);
    });

    this.bias = add(this.bias, multiply(2, deltaArg));

    const deltaWeights = multiply(deltaArg, transpose(this.lastInput));
    const additionalDelta = this.prevDeltaWeights
      ? this.prevDeltaWeights
      : matrix(deltaWeights.map(() => 0));

    this.prevDeltaWeights = multiply(0.05, deltaWeights);

    this.weights = add(
      this.weights,
      subtract(multiply(1.5, deltaWeights), additionalDelta)
    );

    this.parent.train(deltaArg);
  }
}

class Network {
  Layers: Layer[];
  inputLayer: Layer;

  constructor(inputsNumber: number) {
    this.Layers = [];
    this.inputLayer = new Layer(inputsNumber);
  }

  createLayer(neuronsNumber: number) {
    const parentLayer = this.Layers.length
      ? this.Layers[this.Layers.length - 1]
      : this.inputLayer;

    this.Layers.push(new Layer(neuronsNumber, parentLayer));
  }

  calculate(inputData: Matrix) {
    let res = inputData;
    this.Layers.map((layer) => {
      res = layer.calculate(res);
    });

    return res;
  }

  train(data: { input: Matrix; output: Matrix }[]) {
    data.map(({ input, output }) => {
      const outputData = this.calculate(input);
      const delta = subtract(output, outputData);

      this.Layers[this.Layers.length - 1].train(delta);
    });
  }
}

const net = new Network(2);

net.createLayer(4);

net.createLayer(1);

for (let i = 0; i < 1000; ++i) {
  net.train(xorData);
}

console.log(net.calculate(matrix([[0], [0]])));
console.log(net.calculate(matrix([[0], [1]])));
console.log(net.calculate(matrix([[1], [0]])));
console.log(net.calculate(matrix([[1], [1]])));
