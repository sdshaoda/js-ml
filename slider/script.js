import * as tf from '@tensorflow/tfjs'
import * as speechCommands from '@tensorflow-models/speech-commands'

const MODEL_PATH = 'http://127.0.0.1:8080'

let transferRecognizer
let currentIndex = 0

window.onload = async () => {
  const recognizer = speechCommands.create(
    'BROWSER_FFT',
    null,
    MODEL_PATH + '/speech/model.json',
    MODEL_PATH + '/speech/metadata.json'
  )
  await recognizer.ensureModelLoaded()
  transferRecognizer = recognizer.createTransfer('Slider')
  const res = await fetch(MODEL_PATH + '/slider/data.bin')
  const arrayBuffer = await res.arrayBuffer()
  transferRecognizer.loadExamples(arrayBuffer)
  console.log(transferRecognizer.countExamples())
  await transferRecognizer.train({ epochs: 30 })
  console.log('训练完成')
}

window.toggle = async checked => {
  if (checked) {
    await transferRecognizer.listen(
      res => {
        const { scores } = res
        const labels = transferRecognizer.wordLabels()
        const index = scores.indexOf(Math.max(...scores))
        console.log(labels[index])
        play(labels[index])
      },
      {
        overlapFactor: 0,
        probabilityThreshold: 0.75,
      }
    )
  } else {
    transferRecognizer.stopListening()
  }
}

window.play = label => {
  const div = document.querySelector('.slider>div')
  if (label === '上一张') {
    if (currentIndex === 0) {
      return
    }
    currentIndex -= 1
  } else {
    if (currentIndex === document.querySelectorAll('img').length - 1) {
      return
    }
    currentIndex += 1
  }
  div.style.transition = 'transform 1s'
  div.style.transform = `translateX(-${currentIndex * 100}%)`
}
