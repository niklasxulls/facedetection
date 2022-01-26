<?php

require_once 'connect.php';

header("Access-Control-Allow-Origin: *");
header("Content-Type: application/json; charset=UTF-8");

$stmt = "SELECT JSON_ARRAYAGG(JSON_OBJECT('id', users.id, 'name', name, 'first_seen', DATE_FORMAT(first_seen, '%d.%m.%Y'), 'face_images', (SELECT JSON_ARRAYAGG(JSON_OBJECT('path', CONCAT('http://localhost/projects/LogicAIFaceRecProject/backend/api/v1', path), 'points', JSON_OBJECT('x1', x1, 'x2', x2, 'y1', y1, 'y2', y2))) FROM facepaths WHERE uid = users.id))) AS jsonpayload FROM users";
$result = getConn()->query($stmt);

$rows = $result->fetchAll(PDO::FETCH_ASSOC);
$jsonpayload = $rows[0]['jsonpayload'];

if ($jsonpayload != null) {
    echo $jsonpayload;
} else {
    echo json_encode([]);
}