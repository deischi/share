#define NOMINMAX
#include <onnxruntime_cxx_api.h>
#include <dml_provider_factory.h>

#include <stdio.h>
#include <algorithm>

class ThreadingOptions
{
public:
	OrtThreadingOptions* threadingOptions = nullptr;

	ThreadingOptions()
	{
		const OrtApi& ortApi = Ort::GetApi();

		ortApi.CreateThreadingOptions(&threadingOptions);

		ortApi.SetGlobalIntraOpNumThreads(threadingOptions, 0);
		ortApi.SetGlobalInterOpNumThreads(threadingOptions, 0);
	}
	~ThreadingOptions()
	{
		const OrtApi& ortApi = Ort::GetApi();

		ortApi.ReleaseThreadingOptions(threadingOptions);
	}
};


static void ORT_API_CALL LoggingFunction(void* /*param*/, OrtLoggingLevel severity, const char* /*category*/, const char* /*logid*/, const char* /*code_location*/, const char* message)
{
	printf("onnxruntime: %s\n", message);
}

static Ort::Env& getOrtEnv()
{
	static Ort::Env s_OrtEnv{ nullptr };

	if (s_OrtEnv == nullptr)
	{
		ThreadingOptions threadingOptions;

		s_OrtEnv = Ort::Env(threadingOptions.threadingOptions, LoggingFunction, nullptr, ORT_LOGGING_LEVEL_VERBOSE, "");

		s_OrtEnv.DisableTelemetryEvents();
	}

	return s_OrtEnv;
}

typedef unsigned short float16;
__forceinline const float16 f16Float32To16(const float& f)
{
	// This union gives us an easy way to examine and/or set the individual
	// bit-fields of an s23e8.
	union u_u32_s23e8
	{
		unsigned int i;
		float	 f;
	};

	u_u32_s23e8 x;

	x.f = f;

	int e = (x.i >> 23) & 0x000000ff;
	const int s = (x.i >> 16) & 0x00008000;
	int m = x.i & 0x007fffff;

	float16 _h;
	e = e - 127;
	if (e == 128) {
		// infinity or NAN; preserve the leading bits of mantissa
		// because they tell whether it's a signaling of quiet NAN
		_h = (float16)(s | (31 << 10) | (m >> 13));
	}
	else if (e > 15) {
		// overflow to infinity
		_h = (float16)(s | (31 << 10));
	}
	else if (e > -15) {
		// normalized case
		if ((m & 0x00003fff) == 0x00001000) {
			// tie, round down to even
			_h = (float16)(s | ((e + 15) << 10) | (m >> 13));
		}
		else {
			// all non-ties, and tie round up to even
			_h = (float16)(s | ((e + 15) << 10) | ((m + 0x00001000) >> 13));
		}
	}
	else if (e > -25) {
		// convert to subnormal
		m |= 0x00800000; // restore the implied bit
		e = -14 - e; // shift count
		m >>= e; // M now in position but 2^13 too big
		if ((m & 0x00003fff) == 0x00001000) {
			// tie round down to even
		}
		else {
			// all non-ties, and tie round up to even
			m += (1 << 12); // m += 0x00001000
		}
		m >>= 13;
		_h = (float16)(s | m);
	}
	else {
		// zero, or underflow
		_h = (float16)(s);
	}
	return _h;
}


int main()
{

	const OrtApi& ortApi = Ort::GetApi();

	const OrtDmlApi* ortDmlApi = nullptr;
	
	auto sessionOptions = Ort::SessionOptions{};


	Ort::ThrowOnError(ortApi.GetExecutionProviderApi("DML", ORT_API_VERSION, reinterpret_cast<const void**>(&ortDmlApi)));
	Ort::ThrowOnError(ortDmlApi->SessionOptionsAppendExecutionProvider_DML(sessionOptions, 1));

	sessionOptions.DisableMemPattern();
	sessionOptions.DisablePerSessionThreads();
	sessionOptions.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
	sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

	auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

	auto session = Ort::Session(getOrtEnv(), L"..\\Dummy_model.onnx", sessionOptions);

	std::vector<int64_t> dims_input{ 1, 256, 128, 256, 1 };
	std::vector<int64_t> dims_output{ 1, 256, 128, 256, 1 };

	std::vector<Ort::Float16_t> vecInput(256 * 128 * 256);
	for (int i = 0; i < 256 * 128 * 256; i++)
	{
		vecInput[i] = f16Float32To16(float(i % 256) / 256 );
	}

	Ort::Value inputTensor = Ort::Value::CreateTensor<Ort::Float16_t>(memoryInfo, vecInput.data(), vecInput.size(), dims_input.data(), dims_input.size());

	const char* inputName = "input";
	const char* outputName = "conv3d_18";

	auto outputValues = session.Run(Ort::RunOptions{ nullptr }, &inputName, &inputTensor, 1, &outputName, 1);

	const Ort::Float16_t* pOutputData = outputValues[0].GetTensorData<Ort::Float16_t>();
	
	short minOutput = 255;
	short maxOutput = 0;
	for (int i = 0; i < 256 * 128 * 256; i++)
	{
		short r = pOutputData[i];
		minOutput = std::min(minOutput, r);
		maxOutput = std::max(maxOutput, r);
	}

	printf("min/max: %d/%d\n", minOutput,maxOutput);

}


