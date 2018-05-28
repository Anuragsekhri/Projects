<?php


$host = 'pod103.unisonserver.com';
$user = 'netlutio_root';
$pwd = 'root123';
$db = 'netlutio_dsswfly';
$dist = $_GET['dist'];
$dos = $_GET['datesow'];
$doi = $_GET['dateinci'];
$sub =  $_GET['subdate'];
//localhost/my_updated.php?dist=abohar&datesow=2015-09-05&dateinci=2016-04-15&subdate=2016-04-01


$conn = mysqli_connect($host, $user, $pwd, $db);
if(!$conn) {
	die("Error in connection :" . mysqli_connect_error());

}

$response = array();

$sql_query = "select * from Record where dist = '".$dist."' and cdate = '".$dos."'  and wdate > '".$sub."'  and wdate <= '".$doi."'    order by wdate asc" ;
$result = mysqli_query($conn,$sql_query);

if(mysqli_num_rows($result) > 0){

	$response['success']=1;
	$Record = array();
	while($row = mysqli_fetch_assoc($result)) 
	{
		#code...

		array_push($Record,$row);
	}

	$response['Record']=$Record;
}
else 
{
	$response['success'] = 0;
	$response['message'] = 'NO DATA';
}



echo json_encode($response);

mysqli_close($conn);
?>