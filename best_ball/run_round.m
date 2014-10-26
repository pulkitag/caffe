function [] = run_round(prms, baseLr, roundNum, iterCount)

lrPower      = [-2, -1, 0, 1, 2];
prms.maxIter = iterCount + prms.iterPerRound;

for l=1:1:5
	lr = baseLr*power(2,lrPower(l));

	%Save Solver File
	fName = prms.path.solverProtoFile;
	snapPrefix = sprintf(prms.path.snapshot,roundNum,lr); 
	save_solver_proto(prms,fName,lr,snapPrefix);	

	%Save Train File
	logFile = sprintf(prms.path.logFile,roundNum,lr);
	save_train_file(prms,logFile);

	%Run the train file
	system(prms.path.trainFile);	
end

end



