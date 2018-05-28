-- phpMyAdmin SQL Dump
-- version 4.6.5.2
-- https://www.phpmyadmin.net/
--
-- Host: localhost:3307
-- Generation Time: May 28, 2018 at 09:34 AM
-- Server version: 10.1.21-MariaDB
-- PHP Version: 5.6.30

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `ecommerce`
--

-- --------------------------------------------------------

--
-- Table structure for table `data`
--

CREATE TABLE `data` (
  `ID` int(11) NOT NULL,
  `Name` varchar(255) DEFAULT NULL,
  `Price` int(11) DEFAULT NULL,
  `Description` varchar(255) DEFAULT NULL,
  `Category` varchar(255) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `data`
--

INSERT INTO `data` (`ID`, `Name`, `Price`, `Description`, `Category`) VALUES
(1, 'Hp 15-ay2009', 50000, 'Hp Laptop i5 6th gen 8GB Ram 2GB Graphics 1 TB HDD ', 'Laptop'),
(2, 'Dell Inspiron 5000 series ', 75000, 'i7 6th gen dell laptop 16GB Ram 4Gb Graphics 2Tb HDD', 'laptop'),
(3, 'Samsung 36\" LED ', 40000, 'Samsung Tv LED 36\" Full HD HDMI Supported ', 'Tv'),
(4, 'Sony Bravia 40\" 3D Tv', 100000, 'Song Bravia 40\" 3D TV with HDMI Supported ', 'Tv'),
(5, 'Lenovo 181ppq-2', 36000, 'Lenovo i3 5th gen 4GB Ram 1Gb Graphics  500GB HDD', 'Laptop'),
(6, 'Samsung 40\" LED ', 50000, 'Samsung Tv LED 40\" Full HD HDMI Supported ', 'Tv'),
(7, 'Lotus Sunscreen SPF 40', 50, 'SunScreen Lotus SPF 40', 'Cosmetics'),
(8, 'Vasline ColdCream ', 30, 'Cold Cream vasline Cosmetics', 'Cosmetics'),
(9, 'Lotus Sunscreen SPF 30', 30, 'SunScreen Lotus SPF 30', 'Cosmetics'),
(10, 'Apple iPhone 7', 76000, 'Apple iPhone 7 64Gb', 'mobile'),
(11, 'Samsung J7 Prime', 16000, 'samsung mobile j7 prime 32GB Internal  5.5\"  3GB ram', 'mobile'),
(12, 'Samsung J2', 10000, 'samsung mobile j2 8GB Internal  5\"  1GB ram', 'mobile'),
(13, 'Apple iPad ', 7000, 'apple iPad Tab 8Gb internal', 'tab'),
(14, 'Samsung Tab 2821waq-1', 5000, 'Samsung Tab 8Gb internal ', 'tab'),
(15, 'Crocin', 20, 'Crocin Headache Fever Cold Cough Tablet Paracetamol', 'Medicine'),
(16, 'Disprin', 10, 'Disprin Medicine Headache ', 'Medicine'),
(17, 'Genteal Eye Drops', 110, 'Genteal Eye Drops Medicine ', 'Medicine'),
(18, 'SunFlame Non Stick Tawa ', 2000, 'Sunflame Non Stick Tawa Kitchenware', 'KitchenWare'),
(19, 'Sunflame Gas Stove 4 ', 12000, 'Sunflame Gas Stove 4  Kitchenware', 'KitchenWare'),
(20, 'BoroSil Dinner Set', 5000, 'Borosil DinnerSet (2 Large Bowl, 1 Dozen Spoons and forks ,6 Large and 6 Small Plates )', 'KitchenWare'),
(21, 'Apple iPod', 20000, 'Apple iPod Touch 2 GB Memory MP3 supported', 'mobile'),
(22, 'Sony laptop tu281u', 32100, 'Sony Laptop tu281u 8GB ram 2GB graphic ', 'Laptop');

--
-- Indexes for dumped tables
--

--
-- Indexes for table `data`
--
ALTER TABLE `data`
  ADD PRIMARY KEY (`ID`);

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
