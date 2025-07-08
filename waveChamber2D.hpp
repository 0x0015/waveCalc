#pragma once
#include "waveChamber_general.hpp"
#include "syclVector.hpp"

class waveChamber2D{
public:
	static constexpr inline unsigned int dim = 2;
	struct stateVals{
		arrayNDWrapper<double, dim> uVals;
	};
	sycl::vec<double, dim> size;
	sycl::vec<unsigned int, dim> partitions;
	double partitionSize;
	double dt; //delta time
	using chamberDef_t = chamberDef<dim>;
	sycl::queue q{sycl::property::queue::in_order()};
	deviceVector<chamberDef_t> chamberDefs{q};
	simulationEdgeMode edgeMode;
	std::array<stateVals, 3> states;
	std::array<deviceVector<double>, 3> state_dats{q, q, q};
	unsigned int currentStateNum = 0;
	stateVals* currentState = &states[currentStateNum];
	imageWriter imgWriter;
	void init(sycl::vec<double, dim> size, double dt, unsigned int xPartitions, std::span<const chamberDef_t> chambers, simulationEdgeMode edgeMode = REFLECT); //number of partitions for y will be automatially calculated to maintain square partitions
	void step();
	void printuVals();
	void writeRawData();
	void writeToImage(const std::string& filename, double expectedMax);
	void setSinglePoint(sycl::vec<double, dim> point, double val);
	std::vector<std::pair<sycl::vec<double, dim>, std::function<double(double)>>> varryingTimeFuncs;
	void setPointVarryingTimeFunction(sycl::vec<double, dim> point, std::function<double(double)> valWRTTimeGenerator);
	void runSimulation(double time, double imageSaveInterval = -1, double printRuntimeStatisticsInterval = -1, double saveRawDataInterval = -1);
private:
	rawDataWriter rawWriter;
	void initChambers(std::span<const chamberDef_t> chambers);
	void initStateDats();
};

template<> struct waveChamber_t<2>{
	using type = waveChamber2D;
};

