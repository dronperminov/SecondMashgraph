#pragma once

#include "matrix.h"

typedef Matrix<float> GImage;
typedef Matrix<std::tuple<float, float, float>> RGBImage;
typedef std::vector<float> Descriptor;

const bool useHOG = true;
const bool useLBP = true;
const bool useColor = true;

class FeatureExtractor {
	const uint cellCountPerLineHOG = 16; // количество клеток на линию для HOG
	const uint segmentCountHOG = 16; // количество сегментов интервала [-pi, pi] для HOG
	const uint cellCountPerLineLBP = 8; // количество ячеек на линию для LBP
	const uint blocksPerLine = 8; // количество блоков для цветового дескриптора. 8 x 8 = 64 блока на изображение

	// горизонтальное ядро Собеля
	const Matrix<float> sobelX = {
		{ -1, 0, 1 }, 
		{ -2, 0, 2 }, 
		{ -1, 0, 1 }
	};

	// вертикальное ядро Собеля
	const Matrix<float> sobelY = {
		{ 1,  2,  1 }, 
		{ 0,  0,  0 }, 
		{ -1, -2, -1 }
	};

	RGBImage original; // оригинальное RGB изображение
	GImage grayscale; // изображение в оттенках серого
	GImage module; // модуль градиента
	GImage directions; // направление градиента

	uint height; // высота градиента
	uint width; // ширина градиента

	void GetGrayscale(); // получение изображения в оттенках серого
	void GetGradient(); // получение градиента изображения

	void normalizeHistogram(std::vector<float> &histogram); // нормализация гистограммы

	// получение значений ячеек дескрипторов для клетки
	Descriptor calcHistogramHOG(uint i0, uint j0, uint di, uint dj);
	Descriptor calcHistogramLBP(uint i0, uint j0, uint di, uint dj);
	Descriptor getAverageColor(uint i0, uint j0, uint di, uint dj);

	// получение дескрипторов по различным признакам
	Descriptor getHOGdescriptor(); // HOG дескриптор
	Descriptor getLBPdescriptor(); // LBP дескриптор
	Descriptor getColorDescriptor(); // цветовые признаки

public:
	FeatureExtractor(BMP &bmp); // конструктор из BMP изображения

	std::vector<float> Extract(bool hog = useHOG, bool lbp = useLBP, bool color = useColor); // получение дескриптора по признакам
};

#include "featureExtractor.hpp"