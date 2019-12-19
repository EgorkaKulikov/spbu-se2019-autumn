using System.Collections;
using System.Collections.Generic;
using NUnit.Framework;

namespace Task05.FineTests
{
    [TestFixture]
    public class SearchTests
    {
        [TestCase(0)]
        public void Find_ValueInEmptyFineTree_ReturnNull(int key)
        {
            var tree = new BSTFineGrained<int>();

            var actual = tree.Search(key);

            Assert.AreEqual(actual, default(int));
        }

        static (int,int) [] testFound1 = new (int, int)[] {(5,5)};
        static (int,int) [] testFound2 = new (int, int)[] {(5,5), (9,9), (7,7), (8,8)};
        static (int,int) [] testFound3 = new (int, int)[] {(5,5), (9,9), (1,1), (7,7), (6,6), (8,8)};
        static (int,int) [] testFound4 = new (int, int)[] {(1,1), (2,2), (3,3), (4,4), (5,5)};
        static (int,int) [] testNotFound1 = new (int, int)[] {(5,5), (9,9), (1,1), (7,7), (6,6), (8,8)};
        public class MyFactoryClass
        {
            public static IEnumerable TestFoundCases
            {
                get
                {
                    yield return new TestCaseData(testFound1, 5, 5);
                    yield return new TestCaseData(testFound2, 8, 8);
                    yield return new TestCaseData(testFound3, 9, 9);
                    yield return new TestCaseData(testFound4, 5, 5);
                }
            }
            public static IEnumerable TestNotFoundCases
            {
                get
                {
                    yield return new TestCaseData(testNotFound1, 100, 100);
                }
            }
        }
       
        [Test,TestCaseSource(typeof(MyFactoryClass),"TestFoundCases")]
    
        public void Find_ValueInFineGrainedTree_ReturnValue(IEnumerable<(int, int)> elements, int key, int value)
        {
            var fineTree = new BSTFineGrained<int>(elements);

            var actual = fineTree.Search(key);

            Assert.AreEqual(value, actual);
        }

        
        [Test,TestCaseSource(typeof(MyFactoryClass),"TestNotFoundCases")]
        public void Find_NonExistentValueInFineTree_ReturnNull(IEnumerable<(int, int)> elements, int key, int value)
        {
            var fineTree = new BSTFineGrained<int>(elements);

            var actual = fineTree.Search(key);

            Assert.AreEqual(actual, default(int));
        }
    }
}