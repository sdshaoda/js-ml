import * as tf from '@tensorflow/tfjs'
import * as tfvis from '@tensorflow/tfjs-vis'
import { getData } from './data'

window.predict = () => {
  console.log('请等待训练完成后预测')
}

window.onload = async () => {
  const data = getData(400)

  tfvis.render.scatterplot(
    { name: 'XOR 训练集' },
    {
      values: [
        data.filter(p => p.label === 1),
        data.filter(p => p.label === 0),
      ],
    }
  )

  const model = tf.sequential()
  model.add(
    tf.layers.dense({
      units: 4,
      inputShape: [2],
      activation: 'relu',
    })
  )
  model.add(
    tf.layers.dense({
      units: 1,
      activation: 'sigmoid',
    })
  )
  // 欠拟合
  // model.add(
  //   tf.layers.dense({
  //     units: 1,
  //     inputShape: [2],
  //     activation: 'sigmoid',
  //   })
  // )
  model.compile({
    loss: tf.losses.logLoss,
    optimizer: tf.train.adam(0.1),
  })

  const inputs = tf.tensor(data.map(p => [p.x, p.y]))
  const labels = tf.tensor(data.map(p => p.label))

  await model.fit(inputs, labels, {
    epochs: 10,
    callbacks: tfvis.show.fitCallbacks({ name: '训练过程' }, ['loss']),
  })

  // 欠拟合
  // await model.fit(inputs, labels, {
  //   validationSplit: 0.2,
  //   epochs: 200,
  //   callbacks: tfvis.show.fitCallbacks(
  //     { name: '训练过程' },
  //     ['loss', 'val_loss'],
  //     { callbacks: ['onEpochEnd'] }
  //   ),
  // })

  console.log('训练完成')

  window.predict = async form => {
    const pred = await model.predict(
      tf.tensor([[Number(form.x.value), Number(form.y.value)]])
    )
    console.log(pred.dataSync()[0])
  }
}
