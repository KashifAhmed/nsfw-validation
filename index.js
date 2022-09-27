const axios = require('axios') //you can use any http client
const tf = require('@tensorflow/tfjs-node')
const nsfw = require('nsfwjs')
const path = require('path')
const modelPath = path.join(__dirname+'/assets/model_nsfwjs');
console.log('Model path', modelPath)
const images = [
    `https://img.freepik.com/premium-vector/hand-drawn-monkey-ape-vr-box-virtual-nft-style_361671-246.jpg`,
    `https://i.pinimg.com/originals/c4/42/6a/c4426a1484d2fc75e88cca1ce1386581.jpg`,
    `https://i.pinimg.com/originals/dd/f4/4e/ddf44e5f86dbbc840e7ced72e0fcf840.jpg`,
    `https://i.pinimg.com/originals/dd/14/fe/dd14feb4154c6b442d9bf591958c9cf9.png`,
    `https://img.freepik.com/premium-vector/awesome-bored-ape-glass-eye-nft-style_361671-269.jpg`,
    `https://img.freepik.com/premium-vector/hand-drawn-monkey-ape-vr-box-virtual-nft-style_361671-246.jpg`,
    `https://i.pinimg.com/originals/c4/42/6a/c4426a1484d2fc75e88cca1ce1386581.jpg`,
    `https://i.pinimg.com/originals/dd/f4/4e/ddf44e5f86dbbc840e7ced72e0fcf840.jpg`,
    `https://i.pinimg.com/originals/dd/14/fe/dd14feb4154c6b442d9bf591958c9cf9.png`,
    `https://img.freepik.com/premium-vector/awesome-bored-ape-glass-eye-nft-style_361671-269.jpg`,
    `https://img.freepik.com/premium-vector/hand-drawn-monkey-ape-vr-box-virtual-nft-style_361671-246.jpg`,
    `https://i.pinimg.com/originals/c4/42/6a/c4426a1484d2fc75e88cca1ce1386581.jpg`,
    `https://i.pinimg.com/originals/dd/f4/4e/ddf44e5f86dbbc840e7ced72e0fcf840.jpg`,
    `https://i.pinimg.com/originals/dd/14/fe/dd14feb4154c6b442d9bf591958c9cf9.png`,
    `https://img.freepik.com/premium-vector/awesome-bored-ape-glass-eye-nft-style_361671-269.jpg`,
    `https://img.freepik.com/premium-vector/hand-drawn-monkey-ape-vr-box-virtual-nft-style_361671-246.jpg`,
    `https://i.pinimg.com/originals/c4/42/6a/c4426a1484d2fc75e88cca1ce1386581.jpg`,
    `https://i.pinimg.com/originals/dd/f4/4e/ddf44e5f86dbbc840e7ced72e0fcf840.jpg`,
    `https://i.pinimg.com/originals/dd/14/fe/dd14feb4154c6b442d9bf591958c9cf9.png`,
    `https://img.freepik.com/premium-vector/awesome-bored-ape-glass-eye-nft-style_361671-269.jpg`,
    `https://img.freepik.com/premium-vector/hand-drawn-monkey-ape-vr-box-virtual-nft-style_361671-246.jpg`,
    `https://i.pinimg.com/originals/c4/42/6a/c4426a1484d2fc75e88cca1ce1386581.jpg`,
    `https://i.pinimg.com/originals/dd/f4/4e/ddf44e5f86dbbc840e7ced72e0fcf840.jpg`,
    `https://i.pinimg.com/originals/dd/14/fe/dd14feb4154c6b442d9bf591958c9cf9.png`,
    `https://img.freepik.com/premium-vector/awesome-bored-ape-glass-eye-nft-style_361671-269.jpg`,
    `https://img.freepik.com/premium-vector/hand-drawn-monkey-ape-vr-box-virtual-nft-style_361671-246.jpg`,
    `https://i.pinimg.com/originals/c4/42/6a/c4426a1484d2fc75e88cca1ce1386581.jpg`,
    `https://i.pinimg.com/originals/dd/f4/4e/ddf44e5f86dbbc840e7ced72e0fcf840.jpg`,
    `https://i.pinimg.com/originals/dd/14/fe/dd14feb4154c6b442d9bf591958c9cf9.png`,
    `https://img.freepik.com/premium-vector/awesome-bored-ape-glass-eye-nft-style_361671-269.jpg`,
    `https://img.freepik.com/premium-vector/hand-drawn-monkey-ape-vr-box-virtual-nft-style_361671-246.jpg`,
    `https://i.pinimg.com/originals/c4/42/6a/c4426a1484d2fc75e88cca1ce1386581.jpg`,
    `https://i.pinimg.com/originals/dd/f4/4e/ddf44e5f86dbbc840e7ced72e0fcf840.jpg`,
    `https://i.pinimg.com/originals/dd/14/fe/dd14feb4154c6b442d9bf591958c9cf9.png`,
    `https://img.freepik.com/premium-vector/awesome-bored-ape-glass-eye-nft-style_361671-269.jpg`,
    `https://img.freepik.com/premium-vector/hand-drawn-monkey-ape-vr-box-virtual-nft-style_361671-246.jpg`,
    `https://i.pinimg.com/originals/c4/42/6a/c4426a1484d2fc75e88cca1ce1386581.jpg`,
    `https://i.pinimg.com/originals/dd/f4/4e/ddf44e5f86dbbc840e7ced72e0fcf840.jpg`,
    `https://i.pinimg.com/originals/dd/14/fe/dd14feb4154c6b442d9bf591958c9cf9.png`,
    `https://img.freepik.com/premium-vector/awesome-bored-ape-glass-eye-nft-style_361671-269.jpg`
]

async function fn(imageLink) {
    var start = new Date();
    const pic = await axios.get(imageLink, {
        responseType: 'arraybuffer',
    })


    
    
    const model = await nsfw.load('file://./assets/model_nsfwjs/', { size: 299 }) // To load a local model, nsfw.load('file://./path/to/model/')
    // Image must be in tf.tensor3d format
    // you can convert image to tf.tensor3d with tf.node.decodeImage(Uint8Array,channels)
    const image = await tf.node.decodeImage(pic.data, 3)
    const predictions = await model.classify(image)
    image.dispose() // Tensor memory must be managed explicitly (it is not sufficient to let a tf.Tensor go out of scope for its memory to be released).

    let result = '';
    predictions.forEach(prediction => {
        result += prediction.className + '->' + (prediction.probability * 100) + '\n'
    })
    console.log("Total Request time", (new Date() - start), 'ms')
    console.log(result)
}


(async()=>{
    for(var i=0; i<images.length; i++){
        console.log("Image", images[i])
        fn(images[i])
    }
})();
