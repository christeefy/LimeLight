const faceFile = document.getElementById('faceFile')
const faceDiv = document.getElementById('faceDiv')
// const videoFile = document.getElementById('videoFile')
// const videoSubmit = document.getElementById('videoSubmit')

faceFile.addEventListener('change', e => {
    e.preventDefault()
    console.log('got the imgs')

    if (!faceFile.files.length) return

    for (i = 0; i < faceFile.files.length; i++) {
        const reader = new FileReader()
        const newFace = document.createElement('img')
        const subjectName = document.createElement('h3')
        reader.onload = function (e) {
            console.log('loaded', e.target)
            newFace.src = e.target.result
            newFace.classList.add("rounded-circle")
            faceDiv.appendChild(newFace)

            subjectName.innerText = 'Person ' + i.toString(10) // Bug
            faceDiv.appendChild(subjectName)
        }
        reader.readAsDataURL(faceFile.files[i])
    }
})