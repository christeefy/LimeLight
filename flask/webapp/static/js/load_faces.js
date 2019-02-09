const faceFile = document.getElementById('faceFile')
const faceDiv = document.getElementById('faceDiv')

faceFile.addEventListener('change', e => {
    e.preventDefault()

    if (!faceFile.files.length) return

    // Create new row
    var rowDiv = document.createElement('div')
        
    for (let i = 0; i < faceFile.files.length ; i++) {
        const reader = new FileReader()
        const subjectDiv = document.createElement('div')
        const newFace = document.createElement('img')
        const subjectName = document.createElement('p')
        const num_col_per_row = faceFile.files.length

        // Clear any existing data
        while (faceDiv.firstChild) {
            faceDiv.removeChild(faceDiv.firstChild)
        }
        faceDiv.style.display = "block"

        
        rowDiv.classList.add("row")
        subjectDiv.classList.add("col-md-" + Math.floor(12 / num_col_per_row))
        
        faceDiv.appendChild(rowDiv)
        rowDiv.appendChild(subjectDiv)
        subjectDiv.appendChild(newFace)
        subjectDiv.appendChild(subjectName)

        reader.onload = function (e) {
            newFace.src = e.target.result
            newFace.classList.add("rounded-portrait")
            // newFace.classList.add("mx-auto")
            
            // faceDiv.appendChild(newFace)
            subjectName.innerText = 'Subject ' + (i+1).toString()
            subjectName.setAttribute("style", "text-align: center")
            // faceDiv.appendChild(subjectName)
            // faceDiv.appendChild(rowDiv)
        }
        reader.readAsDataURL(faceFile.files[i])
    }
})