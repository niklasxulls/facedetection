import * as Core from './core.js'
import * as API from './api_interaction.js'
import * as tf from '@tensorflow/tfjs'
import {io} from 'socket.io-client'



const cameraMsg = document.querySelector('.camera-msg')
const video = document.getElementById('camera')
const modalUser = document.getElementById('modal_add')
const closeModal = document.getElementById('close_modal')
const wrapper = document.getElementById('canvas-wrapper');

const canvas = document.getElementById('canvas')
const ctx = canvas.getContext('2d')
const ORIGIN = 'localhost:/'
let faces = [];
let maxFacesLength = 200
let count = 0
let label = ''
let score = 0.
let recog;
let localizer;
const RECOG_IMAGE_SIZES = [250, 200]
let labels = []

async function init() {
    if (navigator.mediaDevices.getUserMedia) {
        navigator.mediaDevices.getUserMedia({ video: {} })
            .then(stream => {
                video.srcObject = stream
                cameraMsg.style.display = 'none'
                canvas.style.display = 'block';
                wrapper.className += ' show '
            })
            .catch(err => console.log(err))
    }
    video.onplay = () => Core.showNotification('Camera Active', null, 3000)
    video.addEventListener('loadeddata', () => {
        startDetection()
    })

    let socket = io("http://127.0.0.1:5000/", {rememberTransport: false})
    socket.on('refetch', data => {
        console.log("I am supposed to refetch")
        fetchRecogSetLabels(data)
    })
    socket.on('connect', () => {
        console.log("I am connected")
        socket.emit('connected');
    })
    localizer = await tf.loadGraphModel('/localizer/model.json');
}

async function fetchRecogSetLabels(data) {
    recog = await tf.loadLayersModel('http://localhost/projects/Face_Recognition_Ullsperger/LogicAIFaceRecProject/workspace/models/recognition/export/model.json')
    console.log(recog)
    labels = JSON.parse(data).item
    console.log(labels)
}

init()

async function startDetection() {
    try {
        setInterval(() => {
            detect(localizer)
        }, 80)


        //establish socket connection and receive labels from backend
        //labels = ...
    } catch(err) {
        console.log(err)
    }
}


async function detect(net) {

    if (video) {
        let startTime = new Date()
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
  
        const img = tf.browser.fromPixels(video)
        const resized = tf.image.resizeBilinear(img, [640,640])
        const casted = resized.cast('int32')
        const expanded = casted.expandDims(0)
        const obj = await net.executeAsync(expanded)

        let endTime1 = new Date()
        console.log("took ", endTime1 - startTime, "ms to predict")

        const labels = await obj[1].array();
        const scores = await obj[3].array();
        const bbxs = await obj[7].array();

        let endTime2 = new Date()
        console.log("took another", endTime2 - endTime1, "ms to get results")

        
        // if(scores[0][0] > 0.35) {
        //     addFace(expanded, bbxs[0][0], canvas.width, canvas.height)
        // }

        let endTime3 = new Date()
        console.log("took another", endTime3 - endTime2, "ms to add face")

        requestAnimationFrame(()=>{
            drawRect(bbxs[0], labels[0], scores[0], 0.50, canvas.width, canvas.height, ctx)
        }); 

        let endTime4 = new Date()
        console.log("took another", endTime4 - endTime3, "ms to draw rect")
  
        tf.dispose(img)
        tf.dispose(resized)
        tf.dispose(casted)
        tf.dispose(expanded)
        tf.dispose(obj)
  
      }
}

async function detect_label(imgTensor) {
    // const formData = new FormData()

    // formData.append('face', file)

    // let res = await API.detectFace(formData)
    // res = JSON.parse(res)
    // label = res.label;
    // score = res.score;
    if(!recog) {
        label = undefined;
    }
    let expanded = tf.expandDims(imgTensor, 0)
    const pred =  await recog.predict(expanded)
    const pred_val = pred.dataSync();
    console.log(pred_val)
    let max_index = 0;

    pred_val.forEach((el, i) => {
        if(el >= pred_val[i])
            max_index = i;

    })

    label = pred_val[max_index] > 0.65 ? labels.find(l => l.id === max_index).name : undefined 
}

async function  addFace(img, bbx, width, height) {
    const [y,x,height_r,width_r] = bbx
    const end_height = parseInt(height * height_r - y * height)
    const end_width = parseInt(width * width_r - x * width);
    const img_cropped = tf.image.cropAndResize(img, [bbx], [0], [end_width, end_height])

    let img_resized = tf.squeeze(img_cropped, 0).cast('int32');
    img_resized = tf.image.resizeBilinear(img_resized, [RECOG_IMAGE_SIZES[0], RECOG_IMAGE_SIZES[1]]).cast('int32')

    detect_label(img_resized);
    if(!label && faces.length <= maxFacesLength) {
        const canvas = document.createElement('canvas');
        canvas.width = end_width
        canvas.height = end_height
        await tf.browser.toPixels(img_resized, canvas);
        
        const blob = Core.extractCurrentFrame(`temp.jpeg`, canvas.toDataURL('image/jpeg', 1.))
        const file = new File([blob], `temp.jpeg`, { type: 'image/jpeg' })

        faces.push(file); 
        if(faces.length === maxFacesLength) {
            processFaces();
        }
    }
    
    tf.dispose(img_cropped)
    tf.dispose(img_resized)
}

async function processFaces() {
    modalUser.classList += ' show'
}


function drawRect (boxes, classes, scores, threshold, imgWidth, imgHeight, ctx) {
    for(let i=0; i<=boxes.length; i++){
        if(boxes[i] && classes[i] && scores[i]>threshold){
            const [y,x,height,width] = boxes[i]
            const text = classes[i]
            
            ctx.strokeStyle = '#eee'
            ctx.lineWidth = 2
            ctx.fillStyle = 'white'
            ctx.font = '12px Arial'         
            ctx.beginPath()
            let face_score = Math.round(scores[i]*100)/100
            let msg = `Face - ${face_score}`
            if(label) {
                msg = `${label}`
            }
            ctx.fillText(msg, x*imgWidth, y*imgHeight-10)
            const x1 = x*imgWidth
            const y1 = y*imgHeight
            ctx.rect(x1, y1, width*imgWidth - x1, height*imgHeight -y1);
            ctx.stroke()
        }
    }
}




document.getElementById('submit_face').addEventListener('click', () => {
    const input = document.querySelector('#name_input')
    const formData = new FormData()

    for (let i = 0; i < faces.length; i++) {
        faces[i] = new File([faces[i]], `${i + 1}.jpeg`, { type: 'image/jpeg' })
        formData.append('faces[]', faces[i])
    }
    formData.append('name', input.value)
 
    API.addUserDB(formData)
    closeUserModal()
})

closeModal.addEventListener('click', () => {
    closeUserModal()
})

function closeUserModal() {
    modalUser.className = modalUser.className.replace('show', '')
}