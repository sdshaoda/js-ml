import * as tf from '@tensorflow/tfjs'
// https://github.com/tensorflow/tfjs-models/tree/master/speech-commands
import * as speechCommands from '@tensorflow-models/speech-commands'

const MODEL_PATH = 'http://127.0.0.1:8080/speech'

window.onload = async () => {
  // 识别器
  const recognizer = speechCommands.create(
    'BROWSER_FFT', // 傅立叶变换
    null,
    MODEL_PATH + '/model.json',
    MODEL_PATH + '/metadata.json'
  )

  await recognizer.ensureModelLoaded()

  const labels = recognizer.wordLabels().slice(2)
  console.log(labels)

  const resultEl = document.querySelector('#result')
  resultEl.innerHTML = labels.map(label => `<div>${label}</div>`).join('')

  recognizer.listen(
    res => {
      const { scores } = res
      const maxValue = Math.max(...scores)
      const index = scores.indexOf(maxValue) - 2
      resultEl.innerHTML = labels
        .map(
          (l, i) =>
            `<div style="background: ${i === index && 'red'}">${l}</div>`
        )
        .join('')
      console.log(labels[index])
    },
    {
      overlapFactor: 0.3,
      probabilityThreshold: 0.8,
    }
  )
}
