const API_BASE = 'http://localhost/projects/LogicAIFaceRecProject/backend/api/v1/'

// basic api interaction
export const getUsersDB = async () => await call(API_BASE)

export const getUserDB = async () => await call(`${API_BASE}user`)

export function addUserDB(formData) {
    console.log(formData)
     call(`${API_BASE}add`, {
        body: formData,
        method: 'post'
    });
}

export async function detectFace(formData) {
    return await call(`${API_BASE}detect`, {
        body: formData,
        method: 'post'
    });
}

export async function updateUserNameDB(id, name) {
    const options = { method: 'put', body: { id: id, name: name } }
    return await call(`${API_BASE}update`, options)
}

export async function deleteUserDB(id) {
    const options = { method: 'delete', body: id }
    return await call(`${API_BASE}delete`, options)
}

// base call function
async function call(endpoint, options) {
    if(!options) options = []
    options['no-cors'] = true
    const resp = await fetch(endpoint, options)
    const data = await resp.json()
    return data
}