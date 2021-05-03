import * as tf from '@tensorflow/tfjs'
import * as tfvis from '@tensorflow/tfjs-vis'

window.onload = async () => {
  // 训练集
  const xs = [1, 2, 3, 4]
  const ys = [1, 3, 5, 7]

  // 可视化训练集
  tfvis.render.scatterplot(
    {
      name: '线性回归训练集',
    },
    {
      values: xs.map((x, i) => ({
        x,
        y: ys[i],
      })),
    },
    {
      xAxisDomain: [0, 5],
      yAxisDomain: [0, 8],
    }
  )

  // 定义模型结构
  const model = tf.sequential()
  model.add(
    tf.layers.dense({
      units: 1,
      inputShape: [1],
    })
  )
  model.compile({
    // 损失函数 均方误差
    loss: tf.losses.meanSquaredError,
    // 优化器 随机梯度下降
    optimizer: tf.train.sgd(0.1),
  })

  // 训练模型
  const inputs = tf.tensor(xs)
  const labels = tf.tensor(ys)
  await model.fit(inputs, labels, {
    batchSize: 4,
    epochs: 100,
    callbacks: tfvis.show.fitCallbacks({ name: '训练过程' }, ['loss']),
  })

  // 预测
  const output = model.predict(tf.tensor([5]))
  output.print()
  console.log(output.dataSync()[0])
}
