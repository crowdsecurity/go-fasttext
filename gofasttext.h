#ifndef GO_FASTTEXT_H
#define GO_FASTTEXT_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct Prediction {
	char *label;
	float score;
} Prediction;
 
void fasttext_LoadModel(const char *);
void fasttext_Test(const char *, int, float);
Prediction *fasttext_Predict(const char *, int, float, int *);

#ifdef __cplusplus
}
#endif

#endif // GO_FASTTEXT_H