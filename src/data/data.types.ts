import { Matrix } from "mathjs";

export type TruthyData = TruthyDataItem[];

type TruthyDataItem = {
  input: Matrix;
  output: Matrix;
};
