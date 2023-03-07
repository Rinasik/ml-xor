import { matrix } from "mathjs";
import { TruthyData } from "../data.types";

export const xorData: TruthyData = [
  { input: matrix([[0], [0]]), output: matrix([0]) },
  { input: matrix([[1], [0]]), output: matrix([1]) },
  { input: matrix([[0], [1]]), output: matrix([1]) },
  { input: matrix([[1], [1]]), output: matrix([0]) },
];
