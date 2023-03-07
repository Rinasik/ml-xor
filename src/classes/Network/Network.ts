import { forEach, Matrix, subtract } from "mathjs";
import { Layer } from "../Layer";
import { INetwork } from "./Network.types";

export class Network implements INetwork {
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
    const lastLayer = this.Layers.length - 1;

    data.map(({ input, output }) => {
      const outputData = this.calculate(input);
      let delta = subtract(output, outputData);

      this.Layers.forEach((_, index, arr) => {
        delta = arr[lastLayer - index].train(delta);
      });
    });
  }
}
