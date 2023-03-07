import {
  add,
  Matrix,
  matrix,
  multiply,
  subtract,
  transpose,
  zeros,
} from "mathjs";
import { activation_sigmoid, derivative_sigmoid } from "../../utils/sigmoid";

export class Layer {
  weights: Matrix;
  bias: Matrix;

  neuronsNumber: number;
  parent: Layer | null;
  children: Layer | null;

  lastArguments: Matrix;
  lastInput: Matrix;
  lastDeltaWeights: Matrix | null;

  constructor(neuronsNumber: number, parent?: Layer) {
    this.neuronsNumber = neuronsNumber;
    this.children = null;
    this.lastDeltaWeights = null;

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

    const deltaArg = multiply(
      this?.children?.weights ? transpose(this.children.weights) : 1,
      delta
    ).map((elem, pos) => {
      return elem * derivativeData.get(pos);
    });

    this.bias = add(this.bias, multiply(2, deltaArg));

    const deltaWeights = multiply(deltaArg, transpose(this.lastInput));
    const additionalDelta = this.lastDeltaWeights
      ? this.lastDeltaWeights
      : matrix(deltaWeights.map(() => 0));

    this.lastDeltaWeights = multiply(0.05, deltaWeights);

    this.weights = add(
      this.weights,
      subtract(multiply(1.5, deltaWeights), additionalDelta)
    );

    return deltaArg;
  }
}
