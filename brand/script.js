import * as tf from '@tensorflow/tfjs'
import * as tfvis from '@tensorflow/tfjs-vis'
import { getInputs } from './data'
import { img2x, file2img } from './utils'

const MOBILENET_MODEL_PATH =
  'http://127.0.0.1:8080/mobilenet/web_model/model.json'
const NUM_CLASSES = 3
const BRAND_CLASSES = ['android', 'apple', 'windows']

window.predict = () => {
  console.log('请等待训练完成后预测')
}

window.onload = async () => {
  const { inputs, labels } = await getInputs()
  const surface = tfvis
    .visor()
    .surface({ name: '输入示例', styles: { height: 250 } })
  inputs.forEach(imgEl => {
    surface.drawArea.appendChild(imgEl)
  })

  const mobilenet = await tf.loadLayersModel(MOBILENET_MODEL_PATH)
  mobilenet.summary()
  const layer = mobilenet.getLayer('conv_pw_13_relu')
  const truncatedMobilenet = tf.model({
    inputs: mobilenet.inputs,
    outputs: layer.output,
  })
  const model = tf.sequential()
  model.add(
    tf.layers.flatten({
      inputShape: layer.outputShape.slice(1),
    })
  )
  model.add(
    tf.layers.dense({
      units: 10,
      activation: 'relu',
    })
  )
  model.add(
    tf.layers.dense({
      units: NUM_CLASSES,
      activation: 'softmax',
    })
  )
  model.compile({
    loss: 'categoricalCrossentropy',
    optimizer: tf.train.adam(),
  })

  const { xs, ys } = tf.tidy(() => {
    const xs = tf.concat(
      inputs.map(imgEl => truncatedMobilenet.predict(img2x(imgEl)))
    )
    const ys = tf.tensor(labels)
    return { xs, ys }
  })

  model.fit(xs, ys, {
    epochs: 20,
    callbacks: tfvis.show.fitCallbacks({ name: '训练过程' }, ['loss'], {
      callbacks: ['onEpochEnd'],
    }),
  })

  window.predict = async file => {
    const img = await file2img(file)
    document.body.appendChild(img)
    const pred = tf.tidy(() => {
      const x = img2x(img)
      const input = truncatedMobilenet.predict(x)
      return model.predict(input)
    })
    console.log(BRAND_CLASSES[pred.argMax(1).dataSync()[0]])
  }
}
