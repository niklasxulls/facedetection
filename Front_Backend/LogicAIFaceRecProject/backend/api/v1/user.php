<?php

require_once 'connect.php';

header("Access-Control-Allow-Origin: *");
header("Content-Type: application/json; charset=UTF-8");

$id = $_GET['id'];

if (isset($id)) {
    $stmt = "SELECT JSON_ARRAYAGG(JSON_OBJECT('id', users.id, 'name', name, 'first_seen', first_seen, 'face_images', 
            (SELECT JSON_ARRAYAGG(JSON_OBJECT('path', CONCAT('value', path), 'points', JSON_OBJECT('x1', x1, 'x2', x2, 'y1', y1, 'y2', y2))) 
            FROM facepaths WHERE uid = '$id'))) AS jsonpayload FROM users WHERE users.id = '$id'";

    $result = getConn()->query($stmt);

    $rows = $result->fetchAll(PDO::FETCH_ASSOC);
    $jsonpayload = $rows[0]['jsonpayload'];

    if ($jsonpayload != null) {
        echo $jsonpayload;
    } else {
        echo json_encode(
            array('message' => 'No such user in database.')
        );
    }
} else {
    echo json_encode(
        array('message' => 'id param undefined.')
    );
}
