import * as tf from '@tensorflow/tfjs'
import * as tfvis from '@tensorflow/tfjs-vis'
import { getData } from './data'

window.predict = () => {
  console.log('请等待训练完成后预测')
}

window.onload = async () => {
  const data = getData(400)

  tfvis.render.scatterplot(
    { name: '逻辑回归训练集' },
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
      units: 1,
      inputShape: [2],
      activation: 'sigmoid',
    })
  )
  model.compile({
    loss: tf.losses.logLoss,
    optimizer: tf.train.adam(0.1),
  })

  const inputs = tf.tensor(data.map(p => [p.x, p.y]))
  const labels = tf.tensor(data.map(p => p.label))

  await model.fit(inputs, labels, {
    batchSize: 40,
    epochs: 50,
    callbacks: tfvis.show.fitCallbacks({ name: '训练过程' }, ['loss']),
  })
  console.log('训练完成')

  window.predict = form => {
    const pred = model.predict(
      tf.tensor([[Number(form.x.value), Number(form.y.value)]])
    )
    console.log(pred.dataSync()[0])
  }
}
