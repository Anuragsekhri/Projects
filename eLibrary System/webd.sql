-- phpMyAdmin SQL Dump
-- version 4.6.5.2
-- https://www.phpmyadmin.net/
--
-- Host: localhost:3307
-- Generation Time: May 28, 2018 at 09:42 AM
-- Server version: 10.1.21-MariaDB
-- PHP Version: 5.6.30

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `webd`
--

-- --------------------------------------------------------

--
-- Table structure for table `books`
--

CREATE TABLE `books` (
  `code` varchar(100) NOT NULL,
  `name` varchar(1000) NOT NULL,
  `author` varchar(1000) NOT NULL,
  `publisher` varchar(1000) NOT NULL,
  `issuedto` varchar(100) NOT NULL,
  `issuedate` varchar(1000) NOT NULL,
  `duedate` varchar(1000) NOT NULL,
  `subject` varchar(1000) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `books`
--

INSERT INTO `books` (`code`, `name`, `author`, `publisher`, `issuedto`, `issuedate`, `duedate`, `subject`) VALUES
('A100', 'Physics Fundamental', 'S.P. Taneja', 'BPH', '', '', '', 'Physics'),
('B101', 'Concepts of maths', 'R.D Sharma', 'Dhanpat Rai', '11620066', '27/09/2017', '01/10/2017', 'Maths'),
('AP-321', 'Quadratic Equation ', 'B.K Sharma', 'BPB', '11620066', '08/10/2017', '18/10/2017', 'Maths'),
('CH-876', 'Organic Chemistry ', 'R.L. Aggarwal', 'Mahajan Sons', '', '', '', 'Chemistry'),
('Ch-672', 'Inorganic Chemistry', 'M.K Gupta', 'BPB', '', '', '', 'Chemistry'),
('HU-7661', 'Micro-Economics ', 'S.K Sahney', 'BTH Sons', '', '', '', 'Humanities'),
('Hu-875', 'Macro Economics', 'Hamir Hussain', 'Hamir sons', '', '', '', 'Humanities'),
('Phy-876', 'Light-Ray and Optics', 'Harley', 'David and Sons', '', '', '', 'Physics'),
('OT-542', 'Data Structure in C++', 'S.K Srivastva', 'BPB', '', '', '', 'Other'),
('Co-345', 'Web Designing Html', 'Thomas Powell', 'BPB', '', '', '', 'Other');

-- --------------------------------------------------------

--
-- Table structure for table `user`
--

CREATE TABLE `user` (
  `firstname` varchar(200) NOT NULL,
  `lastname` varchar(200) NOT NULL,
  `email` text NOT NULL,
  `branch` varchar(1000) NOT NULL,
  `sem` varchar(100) NOT NULL,
  `password` text NOT NULL,
  `rollno` varchar(500) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `user`
--

INSERT INTO `user` (`firstname`, `lastname`, `email`, `branch`, `sem`, `password`, `rollno`) VALUES
('esha', 'mm', 'esha@gmail.com', 'Computer Science', '3rd', 'test12345678', '11610065'),
('vishal', 'Rastogi', 'vishal@test.com', 'Computer Science', '3rd', 'test12345678', '11610528'),
('ayush', 'raj', 'ayush@example.com', 'Computer Science', '3rd', 'test1234567890', '11610571'),
('Shivam', 'Gupta', 'shivi98g@gmail.com', 'Information Technology', '3', 'Ludhiana@123', '11620066');

--
-- Indexes for dumped tables
--

--
-- Indexes for table `user`
--
ALTER TABLE `user`
  ADD PRIMARY KEY (`rollno`);

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
