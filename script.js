const URL = "https://teachablemachine.withgoogle.com/models/yeJj93Mq5/";

    let model, webcam, ctx, labelContainer, maxPredictions;
    
    async function init() {

        const modelURL = URL + "model.json";

        const metadataURL = URL + "metadata.json";

 

        // load the model and metadata

     

        model = await tmPose.load(modelURL, metadataURL);

        maxPredictions = model.getTotalClasses();

 

        // Convenience function to setup a webcam

        const size = 600;

        const flip = true; // whether to flip the webcam

        webcam = new tmPose.Webcam(size, size, flip); // width, height, flip

        await webcam.setup(); // request access to the webcam

        await webcam.play();

        window.requestAnimationFrame(loop);

 

        // append/get elements to the DOM

        const canvas = document.getElementById("canvas");

        canvas.width = 500; canvas.height = 550;

        ctx = canvas.getContext("2d");

        labelContainer = document.getElementById("label-container");

        for (let i = 0; i < maxPredictions; i++) { // and class labels

            labelContainer.appendChild(document.createElement("div"));

        }

    }

 

    async function loop(timestamp) {

        webcam.update(); // update the webcam frame

        await predict();

        window.requestAnimationFrame(loop);

    }

 

    async function predict() {

        // Prediction #1: run input through posenet

        // estimatePose can take in an image, video, or canvas html element

        const { pose, posenetOutput } = await model.estimatePose(webcam.canvas);

        // Prediction 2: run input through teachable machine classification model

        const prediction = await model.predict(posenetOutput);

   

        for (let i = 0; i < maxPredictions; i++) {

            const classPrediction = prediction[i].className;

            const probability = prediction[i].probability.toFixed(2) * 100; // Convert probability to percentage

   

            // Update the label with class prediction and probability

            labelContainer.childNodes[i].innerHTML = `${classPrediction}: ${probability}%`;

   

            // Create a visual bar

            const bar = document.createElement('div');

            bar.style.width = `${probability}%`;

            bar.style.height = '20px';  // Adjust the height of the bar
     
   

            // Set the bar color based on class (assuming the first class should be red)

            if (i === 0) {

                bar.style.backgroundColor = 'green';
                bar.style.borderRadius = '30px';
                

            } else {

                bar.style.backgroundColor = 'red';  // Color other classes as red
                bar.style.borderRadius = '30px';

            }

   

            // Append the bar to the labelContainer

            labelContainer.childNodes[i].appendChild(bar);

        }

   

        // finally draw the poses

        drawPose(pose);

    }

    function drawPose(pose) {

        if (webcam.canvas) {

            ctx.drawImage(webcam.canvas, 0, 0);

            // draw the keypoints and skeleton

            if (pose) {

                const minPartConfidence = 0.5;

                tmPose.drawKeypoints(pose.keypoints, minPartConfidence, ctx);

                tmPose.drawSkeleton(pose.keypoints, minPartConfidence, ctx);

            }

        }

    }