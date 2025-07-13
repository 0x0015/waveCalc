#include "waveChamber_general.hpp"
#include "../syclVector.hpp"

template<typename float_t> class waveChamber3D{
public:
	static constexpr inline unsigned int dim = 3;
	struct stateVals{
		arrayNDWrapper<float_t, dim> uVals;
	};
	sycl::vec<float_t, dim> size;
	sycl::vec<unsigned int, dim> partitions;
	float_t partitionSize;
	float_t dt; //delta time
	using chamberDef_t = chamberDef<float_t, dim>;
	sycl::queue q{sycl::property::queue::in_order()};
	deviceVector<chamberDef_t> chamberDefs{q};
	simulationEdgeMode edgeMode;
	std::array<stateVals, 3> states;
	std::array<deviceVector<float_t>, 3> state_dats{q, q, q};
	unsigned int currentStateNum = 0;
	stateVals* currentState = &states[currentStateNum];
	void init(sycl::vec<float_t, dim> size, float_t dt, unsigned int xPartitions, std::span<const chamberDef_t> chambers, simulationEdgeMode edgeMode = REFLECT); //number of partitions for y will be automatially calculated to maintain square partitions
	void step();
	void printuVals();
	void writeRawData();
	void setSinglePoint(sycl::vec<float_t, dim> point, float_t val);
	std::vector<std::pair<sycl::vec<float_t, dim>, std::function<float_t(float_t)>>> varryingTimeFuncs;
	void setPointVarryingTimeFunction(sycl::vec<float_t, dim> point, std::function<float_t(float_t)> valWRTTimeGenerator);
	void runSimulation(float_t time, float_t printRuntimeStatisticsInterval = -1, float_t saveRawDataInterval = -1);
private:
	rawDataWriter<float_t> rawWriter;
	void initChambers(std::span<const chamberDef_t> chambers);
	void initStateDats();
};

template<typename T> struct waveChamber_t<T, 3>{
	using type = waveChamber3D<T>;
};

