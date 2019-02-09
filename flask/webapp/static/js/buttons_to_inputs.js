const faceUploadButton = document.getElementById("faceUploadButton")
const videoUploadButton = document.getElementById("videoUploadButton")
const submitFormButton = document.getElementById("submitFormButton")

faceUploadButton.addEventListener("click", e => {
    document.getElementById("faceFile").click();
    return false
})

videoUploadButton.addEventListener("click", e => {
    // getData
    document.getElementById("videoFile").click();
})

submitFormButton.addEventListener("click", e => {
    document.getElementById("formSubmit").click();
})