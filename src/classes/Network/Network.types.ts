import { Matrix } from "mathjs";
import { Layer } from "../Layer";

export interface INetwork {
  Layers: Layer[];
  inputLayer: Layer;
  createLayer: (neuronsNumber: number) => void;
  calculate: (data: Matrix) => Matrix;
}
