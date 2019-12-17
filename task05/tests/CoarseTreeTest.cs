using System;
using Xunit;

namespace Task05
{
    public class CoarseTreeTest: AbstractTreeTest
    {
        protected override ITree<Int32, Int32> CreateTree() => new CoarseTree<Int32, Int32>();
    }
}
