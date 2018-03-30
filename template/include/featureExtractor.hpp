// получение изображения в оттенках серого
void FeatureExtractor::GetGrayscale() {
	grayscale = GImage(original.n_rows, original.n_cols); // создаём изображение такого же размера

	for (uint i = 0; i < original.n_rows; i++) 
		for (uint j = 0; j < original.n_cols; j++) 
			grayscale(i, j) = 0.299 * std::get<0>(original(i, j)) + 0.587 * std::get<1>(original(i, j)) + 0.114 * std::get<2>(original(i, j)); // каждый пиксель переводим в яркость: Y = 0.299R + 0.587G + 0.114B
}

void FeatureExtractor::GetGradient() {
	uint n = grayscale.n_rows;
	uint m = grayscale.n_cols;

	height = n + (n % cellCountPerLineHOG ? cellCountPerLineHOG - n % cellCountPerLineHOG : 0);
	width = m + (m % cellCountPerLineHOG ? cellCountPerLineHOG - m % cellCountPerLineHOG : 0);

	/********************************* mirror *********************************/
	GImage tmp(n + 2, m + 2); // создаём временное изображение большее с каждой стороны на 2 пикселя

	// зеркально добавляем пиксели к изображению по краям, чтобы после свёртки размер итогового изображения совпадал с первоначальным
	for (uint i = 0; i < n + 2; i++) {
		for (uint j = 0; j < m + 2; j++) {
			uint x = j, y = i;

			if (i < 1)
				y = 1;
			else if (i > n)
				y = n;

			if (j < 1)
				x = 1;
			else if (j > m)
				x = m;

			tmp(i, j) = grayscale(y - 1, x - 1);
		}
	}
	/**************************************************************************/
	
	module = GImage(height, width);
	directions = GImage(height, width);

	for (uint y = 1; y < height + 1; y++) {
		for (uint x = 1; x < width + 1; x++) {
			float sumX = 0, sumY = 0; // значения свёртки по горизонтали и вертикали

			// выполняем свёртку с обоими фильтрами Собеля
			for (uint i = 0; i < 3; i++) {
				for (uint j = 0; j < 3; j++) {
					float v = (y - 1 + i < n && x - 1 + j < m) ? tmp(y - 1 + i, x - 1 + j) : 0;

					sumX += v * sobelX(i, j); // суммируем значение свёртки с горизонтальным ядром Собеля
					sumY += v * sobelY(i, j); // и значение свёртки с вертикальным ядром Собеля
				}
			}

			module(y - 1, x - 1) = std::sqrt(sumX * sumX + sumY * sumY); // модуль градиента есть корень из суммы квадратов значений свёртки
			directions(y - 1, x - 1) = 0.5f + atan2(sumY, sumX) / (2 * M_PI); // in [0, 1]
		}
	}
}

// нормализация гистограммы: все значения переносятся в интервал [0..1]
void FeatureExtractor::normalizeHistogram(Descriptor &histogram) {
	float sum = 0; // переменная для суммирования квадратов значений гистограммы

	for (uint i = 0; i < histogram.size(); i++)
		sum += histogram[i] * histogram[i]; // находим сумму квадратов всех значений

	if (sum < 0.000001)
		return; // если сумма оказалась довольно маленькой, то нормировка не требуется

	sum = std::sqrt(sum); // иначе находим корень из суммы квадратов

	for (uint i = 0; i < histogram.size(); i++)
		histogram[i] /= sum; // и делим каждое значение гистограммы на нормировочное
}

/******************************************************************************************************************************************/
/*                                                             HOG descriptor                                                             */
/******************************************************************************************************************************************/
Descriptor FeatureExtractor::calcHistogramHOG(uint i0, uint j0, uint di, uint dj) {
	Descriptor histogram(segmentCountHOG, 0.0f);

	for (uint i = 0; i < di; i++) {
		for (uint j = 0; j < dj; j++) {
			uint index = uint(directions(i0 + i, j0 + j) * segmentCountHOG) % segmentCountHOG;
			histogram[index] += module(i0 + i, j0 + j);
		}
	}

	normalizeHistogram(histogram);

	return histogram;
}

Descriptor FeatureExtractor::getHOGdescriptor() {
	Descriptor HOG;

	uint di = height / cellCountPerLineHOG;
	uint dj = width / cellCountPerLineHOG;

	for (uint i = 0; i < height; i += di) {
		for (uint j = 0; j < width; j += dj) {
			Descriptor histogram = calcHistogramHOG(i, j, di, dj);

			HOG.insert(HOG.end(), histogram.begin(), histogram.end());
		}
	}

	return HOG;
}

/******************************************************************************************************************************************/
/*                                                             LBP descriptor                                                             */
/******************************************************************************************************************************************/
Descriptor FeatureExtractor::calcHistogramLBP(uint i0, uint j0, uint di, uint dj) {
	Descriptor histogram(256, 0.0f);

	for (uint i = 0; i < di; i++) {
		if (i0 + i == 0 || i0 + i > grayscale.n_rows - 2)
				continue;

		for (uint j = 0; j < dj; j++) {
			if (j0 + j == 0 || j0 + j > grayscale.n_cols - 2)
				continue;

			float value = grayscale(i0 + i, j0 + j);
			uint8_t v = 0;

			// формируем байт в соответствии с pdf'кой
			v |= (value <= grayscale(i0 + i - 1, j0 + j - 1)) << 0;
			v |= (value <= grayscale(i0 + i - 1, j0 + j)) << 1;
			v |= (value <= grayscale(i0 + i - 1, j0 + j + 1)) << 2;
			v |= (value <= grayscale(i0 + i, j0 + j - 1)) << 3;
			v |= (value <= grayscale(i0 + i, j0 + j + 1)) << 4;
			v |= (value <= grayscale(i0 + i + 1, j0 + j - 1)) << 5;
			v |= (value <= grayscale(i0 + i + 1, j0 + j)) << 6;
			v |= (value <= grayscale(i0 + i + 1, j0 + j + 1)) << 7;

			histogram[v]++; // увеличиваем число значений в гистограмме по заданному значению
		}
	}

	normalizeHistogram(histogram);

	return histogram;
}

Descriptor FeatureExtractor::getLBPdescriptor() {
	uint n = grayscale.n_rows + (grayscale.n_rows % cellCountPerLineLBP ? cellCountPerLineLBP - grayscale.n_rows % cellCountPerLineLBP : 0);
	uint m = grayscale.n_cols + (grayscale.n_cols % cellCountPerLineLBP ? cellCountPerLineLBP - grayscale.n_cols % cellCountPerLineLBP : 0);

	Descriptor LBP;

	uint di = n / cellCountPerLineLBP;
	uint dj = m / cellCountPerLineLBP;

	for (uint i = 0; i < n; i += di) {
		for (uint j = 0; j < m; j += dj) {
			Descriptor histogram = calcHistogramLBP(i, j, di, dj);

			LBP.insert(LBP.end(), histogram.begin(), histogram.end());
		}
	}

	return LBP;
}

/******************************************************************************************************************************************/
/*                                                             Color features                                                             */
/******************************************************************************************************************************************/
Descriptor FeatureExtractor::getAverageColor(uint i0, uint j0, uint di, uint dj) {
	Descriptor color(3, 0.0);

	uint count = 0;

	for (uint i = 0; i < di; i++) {
		if (i0 + i >= original.n_rows)
			continue;

		for (uint j = 0; j < dj; j++) {
			if (j0 + j >= original.n_cols)
				continue;

			color[0] += std::get<0>(original(i0 + i, j0 + j));
			color[1] += std::get<1>(original(i0 + i, j0 + j));
			color[2] += std::get<2>(original(i0 + i, j0 + j));
			count++;
		}
	}

	if (count)
		for (uint i = 0; i < 3; i++)
			color[i] /= count * 255.0;

	return color;
}


Descriptor FeatureExtractor::getColorDescriptor() {
	uint n = original.n_rows + (original.n_rows % blocksPerLine ? blocksPerLine - original.n_rows % blocksPerLine : 0);
	uint m = original.n_cols + (original.n_cols % blocksPerLine ? blocksPerLine - original.n_cols % blocksPerLine : 0);

	uint di = n / blocksPerLine;
	uint dj = m / blocksPerLine;

	Descriptor colorDescriptor;

	for (uint i = 0; i < n; i += di) {
		for (uint j = 0; j < m; j += dj) {
			Descriptor avg = getAverageColor(i, j, di, dj);
			colorDescriptor.insert(colorDescriptor.end(), avg.begin(), avg.end());
		}
	}

	return colorDescriptor;
}

FeatureExtractor::FeatureExtractor(BMP &bmp) {
	// получаем оригинальное RGB изображение
	original = RGBImage(bmp.TellHeight(), bmp.TellWidth()); // создаём изображение такого же размера, как и оригинал

	for (int i = 0; i < bmp.TellHeight(); i++) 
		for (int j = 0; j < bmp.TellWidth(); j++) 
			original(i, j) = std::make_tuple(bmp(j, i)->Red, bmp(j, i)->Green, bmp(j, i)->Blue); // запоминаем значения каналов

	GetGrayscale(); // получаем его копию в оттенках серого
	GetGradient(); // находим модуль градиента и его направление
}

Descriptor FeatureExtractor::Extract(bool hog, bool lbp, bool color) {
	Descriptor descrpitor;

	if (hog) {
		Descriptor HOGdescriptor = getHOGdescriptor();
        descrpitor.insert(descrpitor.end(), HOGdescriptor.begin(), HOGdescriptor.end());
	}

	if (lbp) {
		Descriptor LBPdescriptor = getLBPdescriptor();
        descrpitor.insert(descrpitor.end(), LBPdescriptor.begin(), LBPdescriptor.end());
	}

	if (color) {
		Descriptor Colordescriptor = getColorDescriptor();
        descrpitor.insert(descrpitor.end(), Colordescriptor.begin(), Colordescriptor.end());
	}

	return descrpitor;
}