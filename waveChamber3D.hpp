#include "waveChamber_general.hpp"

template<EXECUTION_MODE exMode> class waveChamber3D{
public:
	static constexpr inline unsigned int dim = 3;
	struct stateVals{
		arrayNDWrapper<double, dim> uVals;
	};
	vec<double, dim> size;
	vec<unsigned int, dim> partitions;
	double partitionSize;
	double dt; //delta time
	using chamberDef_t = chamberDef<dim>;
	generalContainer<exMode, chamberDef_t> chamberDefs;
	simulationEdgeMode edgeMode;
	std::array<stateVals, 3> states;
	std::array<generalContainer<exMode, double>, 3> state_dats;
	unsigned int currentStateNum = 0;
	stateVals* currentState = &states[currentStateNum];
	imageWriter imgWriter;
	void init(vec<double, dim> size, double dt, unsigned int xPartitions, std::span<const chamberDef_t> chambers, simulationEdgeMode edgeMode = REFLECT); //number of partitions for y will be automatially calculated to maintain square partitions
	void step();
	void printuVals();
	void writeRawData();
	void writeToImage(const std::string& filename, double expectedMax);
	void setSinglePoint(vec<double, dim> point, double val);
	std::vector<std::pair<vec<double, dim>, std::function<double(double)>>> varryingTimeFuncs;
	void setPointVarryingTimeFunction(vec<double, dim> point, std::function<double(double)> valWRTTimeGenerator);
	void runSimulation(double time, double imageSaveInterval = -1, double printRuntimeStatisticsInterval = -1, double saveRawDataInterval = -1);
private:
	rawDataWriter rawWriter;
	void initChambers(std::span<const chamberDef_t> chambers);
	void initStateDats();
	void calculateBestGpuOccupancy();
	int gpuBlockSize, gpuGridSize, gpuMinGridSize;
};

template<EXECUTION_MODE exMode> struct waveChamber_t<exMode, 3>{
	using type = waveChamber3D<exMode>;
};

