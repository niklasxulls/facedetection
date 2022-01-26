<?php

require_once '../../php_packages/vendor/autoload.php';

use Dotenv\Dotenv;

$dotenv = Dotenv::createImmutable('../../../configs');
$dotenv->load();

function getConn()
{
    static $conn;
    if ($conn === null) {
        $servername = $_ENV['DB_HOST'];
        $username = $_ENV['DB_USER'];
        $password = $_ENV['DB_PASSWORD'];

        try {
            $conn = new PDO("mysql:host=$servername;dbname=$username", $username, $password);
            $conn->setAttribute(PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION);
        } catch (PDOException $e) {
            echo "Connection failed: " . $e->getMessage();
            // remove error print in production => in case of error gives an attacker insight of system
        }
    }
    return $conn;
}
