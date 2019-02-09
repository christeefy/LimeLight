const videoFile = document.getElementById('videoFile')
const videoFeedback = document.getElementById('videoFeedback')

videoFile.addEventListener('change', e => {
    e.preventDefault()
    videoFeedback.innerText = videoFile.files[0].name
})