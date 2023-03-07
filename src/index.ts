import { matrix } from "mathjs";
import { Network } from "./classes/Network";
import { xorData } from "./data/xor/xor.data";

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
