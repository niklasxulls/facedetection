
// make notification appear
export function showNotification(textToShow, color, timeout) {
    let x = document.getElementById('toast')
    x.innerText = textToShow
    if (color) {
        x.style.backgroundColor = color
    }
    x.className = 'show'
    setTimeout(() => x.className = x.className.replace('show', ''), timeout)
}

// returns current canvas frame
export function extractCurrentFrame(name, frame) {
    let blob = dataURItoBlob(frame)
    blob.name = name
    return blob
}

export const getRandomInt = (max) => Math.floor(Math.random() * max);

// data url to blob
function dataURItoBlob(dataURI) {
    let byteString;
    if (dataURI.split(',')[0].indexOf('base64') >= 0)
        byteString = atob(dataURI.split(',')[1]);
    else
        byteString = decodeURIComponent(dataURI.split(',')[1]);

    let mimeString = dataURI.split(',')[0].split(':')[1].split(';')[0];
    let ia = new Uint8Array(byteString.length);
    for (let i = 0; i < byteString.length; i++) {
        ia[i] = byteString.charCodeAt(i);
    }

    return new Blob([ia], { type: mimeString });
}
