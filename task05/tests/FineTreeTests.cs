using System;
using Xunit;

namespace Task05
{
    public class FineTreeTest : ITreeTest
    {
        protected override ITree<Int32, Int32> CreateTree() => new FineTree<Int32, Int32>();
    }
}
