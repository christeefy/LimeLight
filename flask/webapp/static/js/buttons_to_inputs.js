const faceUploadButton = document.getElementById("faceUploadButton")
const videoUploadButton = document.getElementById("videoUploadButton")
const submitFormButton = document.getElementById("submitFormButton")

faceUploadButton.addEventListener("click", e => {
    document.getElementById("faceFile").click();
})

videoUploadButton.addEventListener("click", e => {
    document.getElementById("videoFile").click();
})

submitFormButton.addEventListener("click", e => {
    // document.getElementsByTagName("body")[0].setAttribute("filter", "blur(3px)");
    document.getElementById("formSubmit").click();
})