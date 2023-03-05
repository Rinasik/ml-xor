export const activation_sigmoid = (x) => 1 / (1 + Math.exp(-x));

export const derivative_sigmoid = (x) => {
  const fx = activation_sigmoid(x);

  return fx * (1 - fx);
};
