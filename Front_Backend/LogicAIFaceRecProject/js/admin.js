import * as Core from './core.js'
import * as API from './api_interaction.js'

const users = document.querySelector('.users')
const editUserModal = document.querySelector('.overlay')
const noUsersLabel = document.querySelector('.no-users-label')

function createUser(user) {
    const id = user.id
    const name = user.name
    const first_seen = user.first_seen
    const picture = user.face_images[Core.getRandomInt(5)].path
    const picture2 = user.face_images[Core.getRandomInt(5)].path
    const displayedID = id.substr(0, 7) + '&hellip;'

    const userElement = document.createElement('div')
    userElement.classList.add('user')
    userElement.innerHTML =
        `<img src="${picture}" alt="preview user">
        <div class="info">
            <div class="subitem">
                <p>Face-ID:</p>
                <p>#<span>${displayedID}</span></p>
            </div>
            <div class="subitem">
                <p>Name:</p>
                <span>${name}</span>
            </div>
            <div class="subitem">
                <p>First Seen:</p>
                <span class="date-span">${first_seen}</span>
            </div>
        </div>`

    let btnsElement = document.createElement('div')
    btnsElement.classList.add('btns')

    let settingsBtn = document.createElement('button')
    settingsBtn.classList.add('settings-btn')
    settingsBtn.addEventListener('click', () => settingsUserItem(id, picture2, name))
    settingsBtn.innerHTML = '<img src="https://s2.svgbox.net/materialui.svg?ic=settings&color=fff" width="25" height="25">'

    let deleteBtn = document.createElement('button')
    deleteBtn.addEventListener('click', () => deleteUserItem(userElement, id))
    deleteBtn.innerHTML = '<img src="https://s2.svgbox.net/materialui.svg?ic=delete&color=fff" width="25" height="25">'

    btnsElement.append(settingsBtn)
    btnsElement.append(deleteBtn)
    userElement.append(btnsElement)

    return userElement
}

// get users
async function getUsers() {
    const data = await API.getUsersDB()
    if (data.length > 0) {
        data.forEach(user => users.appendChild(createUser(user)))
        disableLoader()
        Core.showNotification('Users fetched!', 'black', 3000)
    } else {
        disableLoader()
        displayLabel()
    }
}

// update user through settings
function settingsUserItem(id, img, name) {
    editUserModal.classList.toggle('show')
    document.querySelector('#name')
    document.querySelector('.user-img').src = img
    document.querySelector('#user-id').textContent = id
    document.querySelector('#user-name-displayed').textContent = name
}

document.querySelector('#cancel-btn').addEventListener('click', () => editUserModal.classList.toggle('show'))

// delete user
async function deleteUserItem(element, id) {
    const res = await API.deleteUserDB(id)
    if (res.success) {
        element.remove()
        displayLabel()
    }
}

function disableLoader() {
    setTimeout(() => {
        document.getElementById('loader').style.display = 'none'
        document.body.style.overflow = 'visible'
    }, 300);
}

function displayLabel(message) {
    if (!message) {
        if (users.childElementCount <= 0)
            noUsersLabel.style.display = 'block'
        else
            noUsersLabel.style.display = 'none'
    } else {
        noUsersLabel.textContent = message
        noUsersLabel.style.display = 'block'
    }
}

// initial
getUsers()