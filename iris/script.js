import * as tf from '@tensorflow/tfjs'
import * as tfvis from '@tensorflow/tfjs-vis'
import { getIrisData, IRIS_CLASSES } from './data'

window.predict = () => {
  console.log('请等待训练完成后预测')
}

window.onload = async () => {
  const [xTrain, yTrain, xTest, yTest] = getIrisData(0.15)

  const model = tf.sequential()
  model.add(
    tf.layers.dense({
      units: 10,
      inputShape: [xTrain.shape[1]],
      activation: 'sigmoid',
    })
  )
  model.add(
    tf.layers.dense({
      // 最后一层 输出类别为3
      units: 3,
      activation: 'softmax',
    })
  )
  model.compile({
    // 交叉熵损失
    loss: 'categoricalCrossentropy',
    optimizer: tf.train.adam(0.1),
    // 准确度度量
    metrics: ['accuracy'],
  })

  await model.fit(xTrain, yTrain, {
    epochs: 100,
    // 验证集
    validationData: [xTest, yTest],
    callbacks: tfvis.show.fitCallbacks(
      { name: '训练过程' },
      // 损失 验证集损失 准确度 验证集准确度
      ['loss', 'val_loss', 'acc', 'val_acc'],
      { callbacks: ['onEpochEnd'] }
    ),
  })

  console.log('训练完成')

  window.predict = form => {
    const pred = model.predict(
      tf.tensor([
        [
          Number(form.a.value),
          Number(form.b.value),
          Number(form.c.value),
          Number(form.d.value),
        ],
      ])
    )
    console.log(IRIS_CLASSES[pred.argMax(1).dataSync()[0]])
  }
}
