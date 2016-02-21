package suskun.nn;

import java.io.File;
import java.nio.file.Path;
import java.util.Collections;
import java.util.Random;
import java.util.concurrent.*;

public class MultiThreadedStressTest {

    static class TaskResult {
        String id;
        float[][] result;
        long time;

        public TaskResult(String id, float[][] result, long time) {
            this.id = id;
            this.result = result;
            this.time = time;
        }
    }

    static Random random = new Random(1);

    static class ServiceTask implements Callable<TaskResult> {

        QuantizedDnn dnn;
        Path dataPath;

        public ServiceTask(QuantizedDnn dnn, Path path) {
            this.dnn = dnn;
            this.dataPath = path;
        }

        @Override
        public TaskResult call() throws Exception {
            long start = System.currentTimeMillis();
            String id = dataPath.toFile().getName();
            BatchData batchData = BatchData.loadRawBinary(id, dataPath.toFile());
            Collections.shuffle(batchData.getData());
            batchData = new BatchData(id, batchData.getData().subList(0, random.nextInt(batchData.getData().size())));
            float[][] input = batchData.getAsFloatMatrix();
            float[][] result = dnn.calculate(input, 10);
            return new TaskResult(id, result, System.currentTimeMillis() - start);
        }
    }

    public static void multiThreadedStressTest(QuantizedDnn dnn, int threadCount, int taskCount, Path inputFile) throws Exception {
        ExecutorService executor = Executors.newFixedThreadPool(threadCount);
        CompletionService<TaskResult> cs = new ExecutorCompletionService<>(executor);
        for (int i = 0; i < taskCount; i++) {
            cs.submit(new ServiceTask(dnn, inputFile));
        }
        executor.shutdown();
        int c = 0;
        while (c < taskCount) {
            TaskResult result = cs.take().get();
            System.out.println(result.id + " " + result.time);
            c++;
        }
    }

    public static void main(String[] args) throws Exception {
        FuncTest.generateAlignedInput(1000, new File("data/16khz-10s.bin"));
        QuantizedDnn dnn = QuantizedDnn.loadFromFile(new File("data/dnn.extended.tv.model"));
        multiThreadedStressTest(dnn, 8, 1000, new File("data/16khz-10s.bin").toPath());

    }

}
