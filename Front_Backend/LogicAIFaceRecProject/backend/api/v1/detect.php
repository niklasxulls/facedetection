<?php

require_once 'connect.php';

header("Access-Control-Allow-Origin: *");

$id = uniqid('st0u', true);
$faces_img = $_FILES['face'];


$flag = $_SERVER['REQUEST_METHOD'] === 'POST' && isset($faces_img);

if ($flag) {
    $img_path = $faces_img['tmp_name'];

    echo json_encode(contact_python($img_path));

    // echo json_encode(array(
    //     'success' => true,
    //     'message' => 'user successfully created.'
    // ));
    
    
    http_response_code(300);

} else {
    http_response_code(400);
}


function contact_python($img_path) {
    $url = "http://localhost:5000/detect";
    $data = [
        'path' => $img_path,
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


