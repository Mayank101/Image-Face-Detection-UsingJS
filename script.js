const imageUpload = document.getElementById('imageUpload')

const start = async ()=>{
    const container = document.createElement('div')
    container.style.position = 'relative'
    document.body.append(container)
    const labeledFaceDescriptors = await loadLabeledImages()

    //this varibale contains the face matcher data and % required to detect that as a face ie., 0.6 means 60% means 
    //algorithm is 60% sure who that character is
    const faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors, .6)
    document.body.append('Loaded')
    //adding event listener whenever we upload or change the image
    imageUpload.addEventListener('change', async ()=>{
        const image = await faceapi.bufferToImage(imageUpload.files[0])
        container.append(image)
        const canvas = faceapi.createCanvasFromMedia(image)
        container.append(canvas)

        //resize our canvas according to our image
        const displaySize = {width: image.width, height: image.height}
        faceapi.matchDimensions(canvas,displaySize)

        const detections = await faceapi.detectAllFaces(image).withFaceLandmarks().withFaceDescriptors()

        //display detections to the user
        const resizedDetections = faceapi.resizeResults(detections,displaySize)

        //this will go to all the images and find the result which is close to 60%
        const results = resizedDetections.map(d => faceMatcher.findBestMatch(d.descriptor))
        results.forEach((result,i) => {
            const box = resizedDetections[i].detection.box
            const drawBox = new faceapi.draw.DrawBox(box, {label: result.toString()})
            drawBox.draw(canvas)  
        });
    })
}

Promise.all([
    faceapi.nets.faceRecognitionNet.loadFromUri('/models'),
    faceapi.nets.faceLandmark68Net.loadFromUri('/models'),
    faceapi.nets.ssdMobilenetv1.loadFromUri('/models')
]).then(start)



const loadLabeledImages = ()=>{
    //loading all folders inside labels folder
    const labels = ['Black Widow','Captain America','Captain Marvel','Hawkeye','Jim Rhodes','Thor','Tony Stark']
    return Promise.all(
        labels.map(async label =>{

            //array containing all description images
            const descriptions =[]

            //we are running this loop twice as we have 2 images for each character whose face need to be determined
            for(let i=1;i<=2;i++){

                //fetchImage() function only takes images whcih is hosted on the server, 
                //not the local images so we will be using the github link where all these images are hosted
                const img = await faceapi.fetchImage(`https://github.com/Mayank101/Image-Face-Detection-UsingJS/tree/master/labeled_images/${label}/${i}.jpg`)
                const detections = await faceapi.detectSingleFace(img).withFaceLandmarks().withFaceDescriptor()
                descriptions.push(detections.descriptor)
            }
            return new faceapi.LabeledFaceDescriptors(label,descriptions)
        })
    )
}


