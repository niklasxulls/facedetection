<?php

require_once './connect.php';

$id = file_get_contents('php://input');

if ($_SERVER['REQUEST_METHOD'] === 'DELETE' && isset($id)) {
    $rowsaffected = getConn()->exec("DELETE FROM users WHERE id = '$id'");
    if ($rowsaffected > 0) {
        $folder_path = 'data/' . $id;

        if (deleteDirectory($folder_path)) {
            echo json_encode(array(
                'success' => true,
                'message' => 'user successfully deleted.'
            ));
            http_response_code(200);
        } else {
            http_response_code(500);
        }
    } else {
        echo json_encode(array(
            'success' => false,
            'message' => 'no such user in database.'
        ));
        http_response_code(304);
    }
} else {
    http_response_code(400);
}

// delete directory (recursively)
function deleteDirectory($dir)
{
    if (!file_exists($dir)) {
        return true;
    }

    if (!is_dir($dir)) {
        return unlink($dir);
    }

    foreach (scandir($dir) as $item) {
        if ($item == '.' || $item == '..') {
            continue;
        }

        if (!deleteDirectory($dir . DIRECTORY_SEPARATOR . $item)) {
            return false;
        }
    }

    return rmdir($dir);
}
