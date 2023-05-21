using System;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Complex32;

namespace Lab4
{
	class Program
	{
		static void Main(string[] args)
		{
			//float[,] X = {
			//	{ 22.4f, 17.1f,22f},
			//	{224.2f,17.1f,23f },
			//	{151.8f, 14.9f,21.5f },
			//	{147.3f,13.6f, 28.7f},
			//	{152.3f,10.5f,10.2f}
			//};
			//float[,] Y = {
			//		{46.8f,4.4f,11.1f},
			//		{29f,5.5f,6.1f },
			//		{52.1f,4.2f,11.8f},
			//		{37.1f,5.5f,11.9f},
			//		{64f,4.2f,12.9f}
			//}; //Examples
			//float[,] Z1 = { { 75f, 9.6f, 18.5f } };//Examples
			//float[,] Z2 = { { 95f, 12.5f, 16.1f } };

			float[,] X =
			{
			{ 8.1f, 13.4f ,2.25f,8.5f},
				{ 6.9f,14.1f,3.55f,7.2f },
				{ 6.2f,10.7f,1.75f,7.8f },
				{ 8.3f, 12.8f,2.22f,10.5f},
				{ 9.4f,11.3f,1.45f,13.1f },
				{ 9.2f,14.1f,1.85f,9.4f },
				{ 8.5f,11.5f,2.15f,11.3f }
		};
		float[,] Y =
		{
				{4.7f,13.8f,1.2f,10.72f },
				{6.1f,14.0f,1.7f,10.54f },
				{4.3f,11.7f,1.4f,12.83f },
				{ 5.8f,12.0f,1.9f,13.55f},
				{6.3f,14.3f,1.2f,14.67f },
				{5.0f,15.0f,1.4f,15.64f },
				{5.5f,12.9f,1.8f,14.78f }
			};
		float[,] Z1 = { { 7.5f, 10.23f, 2.31f, 10.45f } };//10 var;
		float[,] Z2 = { { 6.5f, 12.02f, 1.36f, 14.41f } };

		float[] Xmid = GetMidMatrix(X);
			float[] Ymid = GetMidMatrix(Y);

			Console.WriteLine("X: ");
			DrawMatrix(X);
			Console.WriteLine("X mid values: ");
			PrintMatrix(Xmid);
			Console.WriteLine("Y: ");
			DrawMatrix(Y);
			Console.WriteLine("Y mid values: ");
			PrintMatrix(Ymid);
			Console.WriteLine("Z1: ");
			DrawMatrix(Z1);
			Console.WriteLine("Z2: ");
			DrawMatrix(Z2);

			float[,] CovMatrixX = CreateCoreletionMatrix(X, Xmid);
			float[,] CovMatrixY = CreateCoreletionMatrix(Y, Ymid);
			Console.WriteLine("X covariative matrix: ");
			DrawMatrix(CovMatrixX);
			Console.WriteLine("Y covariative matrix: ");
			DrawMatrix(CovMatrixY);

			float[,] multipliedXMatrix = MultiplyMatrixToValue(CovMatrixX, X.GetLength(0));
			float[,] multipliedYMatrix = MultiplyMatrixToValue(CovMatrixY, Y.GetLength(0));
			Console.WriteLine("X matrix multiplied by n: ");
			DrawMatrix(multipliedXMatrix);
			Console.WriteLine("Y matrix multiplied by n: ");
			DrawMatrix(multipliedYMatrix);

			float[,] matrixSum = SumMatrix(multipliedXMatrix, multipliedYMatrix);
			Console.WriteLine("Sum of two matrix: ");
			DrawMatrix(matrixSum);

			float[,] unmovedMarkOfUnatedMatrix = MultiplyMatrixToValue(matrixSum, 1f/(X.GetLength(0) + Y.GetLength(0) - 2f));
			Console.WriteLine("Unmoved mark of united matrix: ");
			DrawMatrix(unmovedMarkOfUnatedMatrix);

			float[,] inverseMatrix = GetInveresMatrix(unmovedMarkOfUnatedMatrix);
			Console.WriteLine("Inversed Matrix: ");
			DrawMatrix(inverseMatrix);

			float[] midVectorofCoefficients = GetMidVector(Xmid, Ymid);
			Console.WriteLine("Midle vector of coeficients:");
			PrintMatrix(midVectorofCoefficients);
			float[] markVector = FindPointVector(inverseMatrix, midVectorofCoefficients);
			Console.WriteLine("Vector of marks coefitients of discriminative function: ");
			PrintMatrix(markVector);

			float[] Xmarks =GetDiscriminativeFunc(X, markVector);
			float[] Ymarks = GetDiscriminativeFunc(Y, markVector);
			Console.WriteLine("Marked vectors of discriminative functions Uxi:");
			PrintMatrix(Xmarks);
			Console.WriteLine("Marked vectors of discriminative functions Uyi:");
			PrintMatrix(Ymarks);

			Console.WriteLine("Middle X marks: ");
			Console.WriteLine(GetMidleMark(Xmarks));
			Console.WriteLine("Middle Y marks: ");
			Console.WriteLine(GetMidleMark(Ymarks));

			float disConst = (GetMidleMark(Xmarks) + GetMidleMark(Ymarks)) / 2f;
			Console.WriteLine("Discriminative const: ");
			Console.WriteLine(disConst);

			Console.WriteLine("Mark of Z1 function: ");
			Console.WriteLine(GetMarkOfDisFunc(Z1, markVector));
			Console.WriteLine("Mark of Z2 function: ");
			Console.WriteLine(GetMarkOfDisFunc(Z2, markVector));

			if(GetMarkOfDisFunc(Z1, markVector) < disConst)
			{
				Console.WriteLine("Z1 is belong to second sample");
			}
			else
			{
				Console.WriteLine("Z1 is belong to first sample");
			}

			if(GetMarkOfDisFunc(Z2, markVector) < disConst)
			{
				Console.WriteLine("Z2 is belong to second sample");
			}
			else
			{
				Console.WriteLine("Z2 is belong to first sample");
			}
		}

		public static float GetMarkOfDisFunc(float[,] Z, float[] midleMarks)
		{
			float sum = 0f;

			for(int i =0; i < Z.Length; i++)
			{
				sum += Z[0,i] * midleMarks[i];
			}
			return sum;
		}

		public static float[] GetDiscriminativeFunc(float[,] function, float[] markVectors)
		{
			Matrix matrix = new DenseMatrix(function.GetLength(0), function.GetLength(1));
			for (int i = 0; i < matrix.RowCount; i++)
			{
				for (int j = 0; j < matrix.ColumnCount; j++)
				{
					matrix[i, j] = function[i, j];
				}
			}

			Vector vector = new DenseVector(markVectors.Length);
			for (int i = 0; i < markVectors.Length; i++)
			{
				vector[i] = markVectors[i];
			}

			Vector resVector = (Vector)matrix.Multiply(vector);
			float[] res = new float[resVector.Count];
			for (int i = 0; i < res.Length; i++)
			{
				res[i] = resVector[i].Real;
			}
			return res;
		}


        public static float GetMidleMark(float[] marks)
        {
            float sum = 0f;
            for (int i = 0; i < marks.Length; i++)
            {
                sum += marks[i];
            }
            return sum / marks.Length;
        }

        public static float[] FindPointVector(float[,] inverseMatrix, float[] midVector)
		{
			float[] pointVector = midVector;

			Matrix matrix = new DenseMatrix(inverseMatrix.GetLength(0), inverseMatrix.GetLength(1));
			for (int i = 0; i < matrix.RowCount; i++)
			{
				for (int j = 0; j < matrix.ColumnCount; j++)
				{
					matrix[i, j] = inverseMatrix[i, j];
				}
			}

			Vector vector = new DenseVector(midVector.Length);
			for(int i = 0; i < midVector.Length; i++)
			{
				vector[i] = midVector[i];
			}

			Vector resVector = (Vector)matrix.Multiply(vector);

			for (int i = 0; i < midVector.Length; i++)
			{
				pointVector[i] = resVector[i].Real;
			}

			return pointVector;
		}

		public static float[] GetMidVector(float[] midX, float[] midY)
		{
			float[] res = midX;
			for(int i = 0; i < midX.Length; i++)
			{
				res[i] = midX[i] - midY[i];
			}
			return res;
		}

		public static float[,] GetInveresMatrix(float[,] inputMatrix)
		{
			Matrix matrix = new DenseMatrix(inputMatrix.GetLength(0), inputMatrix.GetLength(1));

			for (int i = 0; i < matrix.RowCount; i++)
			{
				for (int j = 0; j < matrix.ColumnCount; j++)
				{
					matrix[i, j] = inputMatrix[i, j];
				}
			}
			matrix = (Matrix)matrix.Inverse();
			float[,] inverseMatrix = inputMatrix;
			for (int i = 0; i < inverseMatrix.GetLength(0); i++)
			{
				for (int j = 0; j < inverseMatrix.GetLength(1); j++)
				{
					inverseMatrix[i, j] = matrix[i, j].Real;
				}
			}
			return inverseMatrix;
		}

		private static float[,] SumMatrix(float[,] matrix1, float[,] matrix2)
		{
			float[,] res = matrix1;
			for (int i = 0; i < matrix1.GetLength(0); i++)
			{
				for (int j = 0; j < matrix1.GetLength(1); j++)
				{
					res[i, j] += matrix2[i, j]; 
				}
			}
			return res;
		}

		public static float[,] MultiplyMatrixToValue(float[,] matrix, float value)
		{
			float[,] multipliedMatrix = matrix;
			for(int i = 0; i< matrix.GetLength(0); i++)
			{
				for(int j = 0; j< matrix.GetLength(1); j++)
				{
					multipliedMatrix[i, j] = matrix[i, j] * value;
				}
			}
			return multipliedMatrix;
		}

		public static float[] GetMidMatrix(float[,] matrix)
		{
			float[] midValues = new float[matrix.GetLength(1)];
			for(int i = 0; i < matrix.GetLength(1); i++)
			{
				float sum = 0f;
				for(int j = 0; j < matrix.GetLength(0); j++)
				{
					sum += matrix[j, i];
				}
				midValues[i] = sum/matrix.GetLength(0);
			}
			Console.WriteLine();
			return midValues;
		}

		public static float[,] CreateCoreletionMatrix(float[,] matrix, float[] matrixMidel)
		{
			float[,] corelationMatrix = new float[matrix.GetLength(0) + 1,(matrix.GetLength(1) * (matrix.GetLength(1) + 1)) / 2];
			int counter1 = 0;
			int counter2 = 0;
			for (int k = 0; k < corelationMatrix.GetLength(0)-1; k++)
			{
				for (int n = 0; n < corelationMatrix.GetLength(1); n++)
				{
					float matrixNI = matrix[k, counter1];
					float matrixMidI = matrixMidel[counter1];
					float matrixNJ = matrix[k, counter2];
					float matrixMidJ = matrixMidel[counter2];
					corelationMatrix[k, n] = (matrixNI - matrixMidI) * (matrixNJ - matrixMidJ);
					counter2++;
					if (counter2 == matrix.GetLength(1))
					{
						counter1++;
						counter2 = counter1;
					}
				}
				counter1 = 0;
				counter2 = 0;
			}



			for (int k = 0; k < corelationMatrix.GetLength(1); k++)
			{
				float sum = 0f;
				for (int n = 0; n < corelationMatrix.GetLength(0) - 1; n++)
				{
					sum += corelationMatrix[n, k];
				}
				int count = corelationMatrix.GetLength(1) - 1;
				corelationMatrix[corelationMatrix.GetLength(0) - 1, k] = sum / count;
			}
			DrawMatrix(corelationMatrix);

			float[,] res = new float[matrix.GetLength(1), matrix.GetLength(1)];

			int m = 0;
			for(int k = 0; k< res.GetLength(1); k++)
			{
				for(int n = k; n < res.GetLength(1); n++)
				{
					res[k, n] = corelationMatrix[corelationMatrix.GetLength(0) - 1, m];
					res[n, k] = corelationMatrix[corelationMatrix.GetLength(0) - 1, m];
					m++;
				}
			}
			DrawMatrix(res);
			return res;
		}

		public static void DrawMatrix(float[,] matrix)
		{
			int numRows = matrix.GetLength(0);
			int numCols = matrix.GetLength(1);

			for (int row = 0; row < numRows; row++)
			{
				for (int col = 0; col < numCols; col++)
				{
					Console.Write(" {0,12}", matrix[row, col]);
				}
				Console.WriteLine();
			}
			Console.WriteLine();
		}

		public static void PrintMatrix(float[] matrix)
		{
			for (int i = 0; i < matrix.Length; i++)
			{
				Console.Write(" {0,12}", matrix[i]);
			}
			Console.WriteLine("\n");
		}
	}
}
