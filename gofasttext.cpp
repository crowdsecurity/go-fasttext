#include <cstdlib>
#include <sstream>
#include <cstring>
#include "gofasttext.h"
#include "fastText/src/fasttext.h"

//#include "gofasttext.hh"

#ifdef __cplusplus
extern "C"
{
#endif


	static fasttext::FastText *fasttext_instance = NULL;

	void fasttext_LoadModel(const char *modelPath)
	{
		if (fasttext_instance == NULL)
		{
			fasttext_instance = new fasttext::FastText();
		}
		std::string sModelPath;
		sModelPath = modelPath;
		try
		{
			fasttext_instance->loadModel(sModelPath);
		}
		catch(const std::invalid_argument& e)
		{
			std::cerr << e.what() << '\n';
		}
	}

	void fasttext_Test(const char *line, int k, float threshold) {
		if (fasttext_instance == NULL) {
			//TODO
		}


		std::stringstream stream;

		stream << line;

		auto ret = fasttext_instance->test(stream, k, threshold);

		std::cout << std::get<0>(ret) << " | " << std::get<1>(ret) << " | " << std::get<2>(ret) << std::endl;
	}

	Prediction *fasttext_Predict(const char *line, int k, float threshold, int *outputSize) {
		if (fasttext_instance == NULL) {
			//TODO
		}

		std::vector<std::pair<float, std::string>> predictions;

		std::stringstream stream;

		stream << line;

		try {
			fasttext_instance->predictLine(stream, predictions, k, threshold);
		}
		catch (const std::invalid_argument& e) {
			std::cerr << e.what() << '\n';
		}

		*outputSize = predictions.size();

		Prediction *p = (Prediction *)malloc(sizeof(*p) * predictions.size() + 1);

		if (p == NULL) {
			*outputSize = 0;
			//*error = strdup("Unable to allocate memory for predictions");
			return NULL;
		}

		for (auto it = predictions.begin(); it != predictions.end(); it++)
		{
			int i = it - predictions.begin();
			char *cLabel = (char *)malloc(sizeof(*cLabel) * it->second.size());
			if (cLabel == NULL) {
				std::cerr << "Unable to allocate memory for label" << std::endl;
				free(p);
				*outputSize = 0;
				//FIXME: labels that were previously allocated will be dangling
				return NULL;
			}
			cLabel = strcpy(cLabel, it->second.c_str());
			p[i] = Prediction{ cLabel, it->first };
		}

		return p;
	}

#ifdef __cplusplus
}
#endif