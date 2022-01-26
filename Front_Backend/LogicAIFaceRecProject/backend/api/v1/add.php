<?php

require_once 'connect.php';

header("Access-Control-Allow-Origin: *");
// header("Access-Control-Allow-Credentials", "true");
// header("Access-Control-Allow-Methods", "GET,HEAD,OPTIONS,POST,PUT");
// header("Access-Control-Allow-Headers", "Access-Control-Allow-Headers, Origin,Accept, X-Requested-With, Content-Type, Access-Control-Request-Method, Access-Control-Request-Headers");
// header("Content-Type: application/json; charset=UTF-8");

$id = uniqid('st0u', true);
$name = $_POST['name'];
$first_seen = date('Y-m-d H:i:s');
// $coordinates = json_decode($_POST['coordinates']);
$faces_img = $_FILES['faces'];


// $flag = $_SERVER['REQUEST_METHOD'] === 'POST' && isset($faces_img) && isset($coordinates) && isset($name)
$flag = $_SERVER['REQUEST_METHOD'] === 'POST' && isset($faces_img)  && isset($name);

//           && strlen($name) >= 3;
    // && count($faces_img['name']) === count($coordinates) && strlen($name) >= 3;


if ($flag) {
    // database interaction
    // $stmt = getConn()->prepare('INSERT INTO users (id, name, first_seen) VALUES (:id, :name, :first_seen)');
    // $stmt->bindParam(':id', $id);
    // $stmt->bindParam(':name', $name);
    // $stmt->bindParam(':first_seen', $first_seen);
    // $stmt->execute();

    // file/facepaths handling
    $img_path = dirname(__FILE__) . '\\' . handleFileUpload($id, $faces_img, $coordinates);

    contact_python($name, $id, $img_path);


    echo json_encode(array(
        'success' => true,
        'message' => 'user successfully created.'
    ));
    
    
    http_response_code(300);

} else {
    http_response_code(400);
}

// handlers/validators
function handleFileUpload($id, $faces_img, $coordinates)
{
    if (count($faces_img['name']) <= 200 && contentTypeValidation($faces_img['type'])) {
        $folder = 'data\\' . $id;
        if (mkdir($folder, 0755, true)) {
            // getting names fraction of array
            $names = $faces_img['name'];
            $tmp_dsts = $faces_img['tmp_name'];
            $data = array_combine($tmp_dsts, $names);

            $i = 0;
            foreach ($data as $temp_folder => $image_name) {
                $img_path = $folder . '\\' . $image_name;
                // database
                // $points = $coordinates[$i];
                // $stmt = "INSERT INTO facepaths (uid, path, x1, x2, y1, y2) VALUES ('$id', '/$img_path', '$points->x1', '$points->x2', '$points->y1', '$points->y2')";
                // $stmt = "INSERT INTO facepaths (uid, path, x1, x2, y1, y2) VALUES ('$id', '/$img_path', '0', '0', '0', '0')";

                // getConn()->exec($stmt);

                // locally
                move_uploaded_file($temp_folder, $img_path);
                $i++;
            }
            return $folder;

        } else {
            http_response_code(500);
        }
    } else {
        http_response_code(400);
    }
}

function contact_python($name, $id, $img_path) {
    $url = "http://localhost:5000/add";
    $data = [
        'id' => $id,
        'name' => $name,
        'dir' => $img_path
    ];
    $curl = curl_init($url);
    $headers = array(
        "Accept: application/json",
        "Content-Type: application/json",
    );
    curl_setopt($curl, CURLOPT_HTTPHEADER, $headers);
    curl_setopt($curl, CURLOPT_POST, true);
    curl_setopt($curl, CURLOPT_POSTFIELDS, json_encode($data));
    curl_setopt($curl, CURLOPT_RETURNTRANSFER, true);

    $response = curl_exec($curl);
    curl_close($curl);
    return $response;
}

function contentTypeValidation($arrayOfFileExt)
{
    // extensions allowed to upload
    $ext_allowed = array('image/png', 'image/jpeg', 'image/jpg');
    foreach ($arrayOfFileExt as $ext) {
        if (!in_array($ext, $ext_allowed)) {
            return false;
        }
    }
    return true;
}
