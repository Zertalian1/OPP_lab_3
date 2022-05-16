#include <mpi.h>
#include <cstdlib>
#include <iostream>

#define DIMENSION 2


using namespace std;

void fillTables(int* displs, int* recvcounts, int n1, int n3, int N3, int* dims, int procSize) {
    int currentStart = 0;
    for (int i = 1, temp = 0; i < procSize; ++i) {
        if (i % dims[1] != 0) {
            temp = currentStart + (i % dims[1]) * n3;
            displs[i] = temp;                           // заполнение смещений элементов в строке
        }
        else {
            temp = N3 * n1 * (i / dims[1]);             // заполнение смещений строк
            currentStart = temp;
            displs[i] = temp;
        }
        recvcounts[i] = 1;
    }
    displs[0] = 0;
    recvcounts[0] = 1;
}

void fillXandYComms(MPI_Comm* yComms, MPI_Comm* xComms, MPI_Comm comm2d) {
    //                      x       y   
    int remainDimsY[2] = { true,  false };
    int remainDimsX[2] = { false, true };

    MPI_Cart_sub(comm2d, remainDimsX, xComms); // перемещение по x возможно, по y - нет
    MPI_Cart_sub(comm2d, remainDimsY, yComms); // перемещение по y возможно, по x - нет
}

void printMx(double* matrix, int N1, int N2) {
    for (int i = 0; i < N1; ++i) {
        for (int j = 0; j < N2; ++j) {
            cout << matrix[i * N2 + j] << " ";
        }
        cout << endl;
    }
}

void initMxA(double* matrixA, int N1, int N2) {
    for (int i = 0; i < N1; ++i) {
        for (int j = 0; j < N2; ++j) {
            matrixA[i * N2 + j] = i * N2 + j + 1;
        }
    }
}

void initMxB(double* matrixB, int N2, int N3) {
    for (int i = 0; i < N2; ++i) {
        for (int j = 0; j < N3; ++j) {
            matrixB[i * N3 + j] = i * N3 + j + 1;
        }
    }
}

void collectMxParts(double* matrixCPart, int n1, int n3, int N3, double* matrixC, int* displs, int* recvcounts, MPI_Comm& comm2d) {
    /*ситуация аналогичная матрице B в spreadMatrixAandB*/
    MPI_Datatype cMatBlock, cMatBlockType;

    MPI_Type_vector(n1, n3, N3, MPI_DOUBLE, &cMatBlock);
    MPI_Type_commit(&cMatBlock);
    MPI_Type_create_resized(cMatBlock, 0, sizeof(double), &cMatBlockType);
    MPI_Type_commit(&cMatBlockType);

    MPI_Gatherv(matrixCPart, n1 * n3, MPI_DOUBLE, matrixC, recvcounts, displs, cMatBlockType, 0, comm2d);
}

void multiplyMx(const double* matrixAPart, const double* matrixBPart, int n1, int n2, int n3, double* matrixCPart) {
    for (int i = 0; i < n1; ++i) {
        for (int j = 0; j < n3; ++j) {
            matrixCPart[i * n3 + j] = 0;
            for (int k = 0; k < n2; ++k) {
                matrixCPart[i * n3 + j] += matrixAPart[i * n2 + k] * matrixBPart[k * n3 + j];
            }
        }
    }
}

void spreadMxAandB(int N1, int N2, int N3, double* matrixA, double* matrixB, double* matrixAPart, double* matrixBPart, int* dims, MPI_Comm* xComms, MPI_Comm* yComms, int* procCoords) {
    /* Задание типа данных для вертикальной полосы вВ 
     Этот тип создать необходимо, т.к. в языке С массив в памяти
     располагается по строкам. Для массива А такой тип создавать
     нет необходимости, т.к. там передаются горизонтальные полосы,
     а они в памяти расположены непрерывно. */
    MPI_Datatype bMxColumn, bMxColumnType;
    /*Функция MPI_TYPE_VECTOR является более универсальным конструктором, который позволяет 
    реплицировать типы данных в области, состоящие из блоков равного объема. 
    Каждый блок получается конкатенацией некоторого количества копий старого типа. 
    Пространство между блоками кратно размеру old datatype.

    MPI_TYPE_VECTOR(count, blocklength, stride, oldtype, newtype)*/
    MPI_Type_vector(N2, N3 / dims[1], N3, MPI_DOUBLE, &bMxColumn);
    //Операция commit объявляет тип данных, то есть формально описывает коммуникационный буфер, но не содержимое этого буфера.
    MPI_Type_commit(&bMxColumn);
    /*  MPI_TYPE_CREATE_RESIZED(oldtype, lb, extent, newtype)
        IN	oldtype	входной тип данных (дескриптор)
        IN	lb	новая нижная граница типа данных (целое)
        IN	extent	новая длина типа данных (целое)
        OUT	newtype	выходной тип данных (дескриптор)
    */
    MPI_Type_create_resized(bMxColumn, 0, N3 / dims[1] * sizeof(double), &bMxColumnType);
    MPI_Type_commit(&bMxColumnType);

    // распределение матрицы A по (y, 0)
    if (procCoords[1] == 0)
        MPI_Scatter(matrixA, N1 * N2 / dims[0], MPI_DOUBLE, matrixAPart, N1 * N2 / dims[0], MPI_DOUBLE, 0, *yComms);

    // распределение матрицы A по (y, 0) to (y, b)
    MPI_Bcast(matrixAPart, N1 * N2 / dims[0], MPI_DOUBLE, 0, *xComms);

    // распределение матрицы B по (0, x)
    if (procCoords[0] == 0)
        MPI_Scatter(matrixB, 1, bMxColumnType, matrixBPart, N2 * N3 / dims[1], MPI_DOUBLE, 0, *xComms);

    // распределение матрицы A по (0, x) в (a, x)
    MPI_Bcast(matrixBPart, N2 * N3 / dims[1], MPI_DOUBLE, 0, *yComms);
}

int main(int argc, char* argv[]) {
    int procSize=0;

    MPI_Init(&argc, &argv);

    MPI_Comm comm2d;
    MPI_Comm_size(MPI_COMM_WORLD, &procSize);

    // размер решётки
    int dims[DIMENSION] = { 0, 0 };

    int periods[DIMENSION] = { 0, 0 };

    int procCoords[DIMENSION];
    int reorder = 1;
    // Количество элементов, полученных от каждого процесса
    int* recvcounts = new int [procSize];

    // Данные, полученные от процесса j, помещаются в буфер приема смещения корневого 
    // процесса displs[j] элементов из указателя sendbuf.
    int* displs = new int[procSize];

    double startTime = MPI_Wtime();

    

    // определение размеров решетки: dims
    MPI_Dims_create(procSize, 2, dims);

   /* // берём размеры матриц из командной строки
    int N1 = atoi(argv[1]);
    int N2 = atoi(argv[2]);
    int N3 = atoi(argv[3]);

    dims[0] = atoi(argv[4]);
    dims[0] = atoi(argv[5]);
    */

    int N1 = 512;
    int N2 = 512;
    int N3 = 512;

    // создание коммуникатора : comm2d
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &comm2d);

    // получение своих координат в двумерной решетке: coords
    MPI_Cart_get(comm2d, 2, dims, periods, procCoords);

    // коммуникаторы для строк сетки / столбцов
    MPI_Comm yComms[1];
    MPI_Comm xComms[1];

    // вынести в верхний левый узел
    double* matrixA = NULL;
    double* matrixB = NULL;
    double* matrixC = NULL;

    double* matrixAPart = new double[N1 * N2 / dims[0]];
    double* matrixBPart = new double[N2 * N3 / dims[1]];
    double* matrixCPart = new double[N1 / dims[0] * N3 / dims[1]];

    // определение сдвигов для элементов 2 мерной таблицы узлов
    fillTables(displs, recvcounts, N1 / dims[0], N3 / dims[1], N3, dims, procSize);

    if (procCoords[0] == 0 && procCoords[1] == 0) {  // процесс в верхнем левом углу заполняет матрицы
        matrixA = new double[N1 * N2];
        matrixB = new double[N2 * N3];
        matrixC = new double[N1 * N3];
        initMxA(matrixA, N1, N2);
        initMxB(matrixB, N2, N3);
    }
    // создание коммуникаторов для строк сетки / столбцов
    fillXandYComms(yComms, xComms, comm2d);
    // распространение матриц / раздача кусков
    spreadMxAandB(N1, N2, N3, matrixA, matrixB, matrixAPart, matrixBPart, dims, xComms, yComms, procCoords);
    // перемножение матриц
    multiplyMx(matrixAPart, matrixBPart, N1 / dims[0], N2, N3 / dims[1], matrixCPart);
    // сбор всех частей посчитаной матрицы
    collectMxParts(matrixCPart, N1 / dims[0], N3 / dims[1], N3, matrixC, displs, recvcounts, comm2d);

    double endTime = MPI_Wtime();

    if (procCoords[0] == 0 && procCoords[1] == 0) {
        //printMx(matrixC, N1, N3);
        std::cout << "Time taken: " << endTime - startTime;
    }
    if (procCoords[0] == 0 && procCoords[1] == 0) {
        delete[] matrixA;
        delete[] matrixB;
        delete[] matrixC;
    }
    delete[] recvcounts;
    delete[] displs;
    delete[] matrixAPart;
    delete[] matrixBPart;
    delete[] matrixCPart;

    MPI_Finalize();

    return 0;
}